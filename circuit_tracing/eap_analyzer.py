#!/usr/bin/env python3
"""
EAP Analyzer Core Class
Refactored based on original eap_core.py, focused on circuit tracing functionality
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import re


class EAPAnalyzer:
    """
    EAP (Edge Attribution Patching) Analyzer
    
    Features:
    - Capture activation states of model layers
    - Calculate edge importance scores
    - Support answer logit difference metric for modular arithmetic tasks
    - Provide diagnostic functions for debugging
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device

        # Model configuration
        self.num_heads = getattr(self.model.config, 'num_attention_heads',
                                 getattr(self.model.config, 'n_head', None))
        if self.num_heads is None:
            raise ValueError("Cannot determine number of attention heads from model configuration")
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', None)
        print(f"[Init] model={self.model.__class__.__name__}, layers={self.num_layers}, heads={self.num_heads}")

        # Build computation graph
        self.nodes, self.edges = self._define_graph()
        
        # Task mode configuration
        self.task_mode = config.get("task_mode", "modular_arithmetic")
        if self.task_mode == "modular_arithmetic":
            print(f"[Init] task_mode={self.task_mode}, using answer logit difference metric")
        else:
            print(f"[Init] task_mode={self.task_mode}, using default metric")

        # Runtime state
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        # Edge score calculation mode
        self.edge_score_mode = config.get("edge_score_mode", "vjp")
        print(f"[Init] edge_score_mode={self.edge_score_mode}")

        # Diagnostic configuration
        self.debug_layer_for_head_check = int(config.get("debug_layer_for_head_check", config['layers_to_analyze'][-1]))
        self.debug_top_heads = int(config.get("debug_top_heads", min(8, self.num_heads)))
        self.do_hook_check_once = bool(config.get("do_hook_check_once", True))
        self.do_diag_heads_once = bool(config.get("do_diag_heads_once", True))
        self.do_head_ablation_once = bool(config.get("do_head_ablation_once", False))
        print(f"[Init] diagnostics: hook_check={self.do_hook_check_once}, diag_heads={self.do_diag_heads_once}, head_ablation={self.do_head_ablation_once}")

        # Runtime control
        self.temp_disable_ckpt_for_grad = bool(config.get("temp_disable_checkpointing_during_grad", True))
        print(f"[Init] temp_disable_checkpointing={self.temp_disable_ckpt_for_grad}")

        self._printed_hook_check = False
        self._printed_diag_heads = False
        self._printed_ablation = False

    def _define_graph(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Define computation graph structure"""
        nodes, edges = [], []
        layers = self.config['layers_to_analyze']
        num_heads = self.num_heads

        # Add nodes
        nodes.append("tok_embeds")
        for l in layers:
            nodes.append(f"m{l}")
            nodes.append(f"a{l}")  # Entire a{l} block as node
            for h in range(num_heads):
                nodes.append(f"a{l}.h{h}")

        if not layers:
            print(f"[Graph] Defined {len(nodes)} nodes and {len(edges)} edges")
            return nodes, edges

        # Add edges
        edges.append(("tok_embeds", f"m{layers[0]}"))
        for h in range(num_heads):
            edges.append(("tok_embeds", f"a{layers[0]}.h{h}"))

        for i, l_from in enumerate(layers):
            # Intra-layer connections
            for h in range(num_heads):
                edges.append((f"a{l_from}.h{h}", f"m{l_from}"))
            edges.append((f"a{l_from}", f"m{l_from}"))

            # Inter-layer connections
            if i + 1 < len(layers):
                l_to = layers[i + 1]
                edges.append((f"m{l_from}", f"m{l_to}"))
                edges.append((f"a{l_from}", f"m{l_to}"))
                for h_to in range(num_heads):
                    edges.append((f"a{l_from}.h{h_to}", f"m{l_to}"))
                    for h_from in range(num_heads):
                        edges.append((f"m{l_from}", f"a{l_to}.h{h_from}"))
                        edges.append((f"a{l_from}.h{h_to}", f"a{l_to}.h{h_from}"))
        
        print(f"[Graph] Defined {len(nodes)} nodes and {len(edges)} edges")
        return nodes, edges

    def _extract_numbers_from_prompt(self, prompt: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract numbers a, b, c from prompt"""
        match = re.search(r'\((\d+)\s*[^\d]*\s*(\d+)\)\s*[^\d]*\s*(\d+)', prompt)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None

    def _get_answer_logit(self, logits: torch.Tensor, answer: int) -> torch.Tensor:
        """Get logit value for specific answer"""
        last_token_logits = logits[:, -1, :]
        answer_str = str(answer)
        
        # Try multiple encoding methods
        candidates = [answer_str, f" {answer_str}", f"\n{answer_str}"]
        best_logit = None
        best_logit_val = float('-inf')
        
        for candidate in candidates:
            tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(tokens) > 0:
                token_id = tokens[0]
                logit_tensor = last_token_logits[:, token_id].mean()
                logit_val = logit_tensor.item()
                if logit_val > best_logit_val:
                    best_logit_val = logit_val
                    best_logit = logit_tensor
        
        return best_logit if best_logit is not None else torch.tensor(0.0, device=logits.device, requires_grad=True)

    def _save_activation_hook(self, name: str):
        """Create hook to save activations"""
        def hook(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            self.activations[name] = act
        return hook

    def _save_o_proj_preact_hook(self, layer_idx: int):
        """Create hook to save o_proj pre-activation"""
        num_heads = self.num_heads
        def pre_hook(module, inputs):
            x = inputs[0]  # [B, S, H], H = num_heads * head_dim
            self.activations[f"a{layer_idx}"] = x  # Entire block
            bs, sl, hs = x.shape
            assert hs % num_heads == 0, f"Hidden dimension {hs} cannot be divided by number of heads {num_heads}"
            hd = hs // num_heads
            x_reshaped = x.view(bs, sl, num_heads, hd)
            for h in range(num_heads):
                self.activations[f"a{layer_idx}.h{h}"] = x_reshaped[:, :, h, :]  # Only for value extraction/alignment
            return None
        return pre_hook

    def _register_forward_hooks(self):
        """Register forward propagation hooks"""
        self._clear_hooks_and_data()
        
        # Embedding layer
        if hasattr(self.model, 'get_input_embeddings'):
            embed_module = self.model.get_input_embeddings()
        elif hasattr(self.model.model, 'embed_tokens'):
            embed_module = self.model.model.embed_tokens
        else:
            raise ValueError("Cannot find input embedding module in model")
        self.hooks.append(embed_module.register_forward_hook(self._save_activation_hook("tok_embeds")))
        
        # Each layer
        for l in self.config['layers_to_analyze']:
            layer = self.model.model.layers[l]
            self.hooks.append(layer.mlp.register_forward_hook(self._save_activation_hook(f"m{l}")))
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                handle = layer.self_attn.o_proj.register_forward_pre_hook(self._save_o_proj_preact_hook(l))
                self.hooks.append(handle)
            else:
                raise ValueError(f"Layer {l} does not have self_attn.o_proj; cannot register pre-hook for heads")

    def _clear_hooks_and_data(self):
        """Clear hooks and data"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _calculate_metric(self, logits: torch.Tensor, prompt: Optional[str] = None) -> torch.Tensor:
        """Calculate task-related metric"""
        if self.task_mode == "modular_arithmetic":
            if prompt is None:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            # Extract numbers and calculate correct answer
            a, b, c = self._extract_numbers_from_prompt(prompt)
            if a is None or b is None or c is None:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            correct_answer = (a + b) % c
            answer_logit_tensor = self._get_answer_logit(logits, correct_answer)
            return answer_logit_tensor
        else:
            # Default to using mean logits
            return logits[:, -1, :].mean()

    def _slice_head(self, full: torch.Tensor, h: int) -> torch.Tensor:
        """Slice the h-th head from head block"""
        hd = full.shape[-1] // self.num_heads
        return full[:, :, h * hd:(h + 1) * hd]

    @staticmethod
    def _right_align_tuples(*tensors: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        """Right-align tensor tuples"""
        if len(tensors) == 0:
            return tuple()
        valids = [t for t in tensors if (isinstance(t, torch.Tensor) and t.dim() == 3)]
        if not valids:
            return tensors
        min_len = min(t.shape[1] for t in valids)
        aligned = []
        for t in tensors:
            if t is None or not isinstance(t, torch.Tensor) or t.dim() != 3:
                aligned.append(t)
            else:
                if t.shape[1] == min_len:
                    aligned.append(t)
                else:
                    aligned.append(t[:, -min_len:, :])
        return tuple(aligned)

    def _edge_score_vjp_fallback(self,
                                 src_name: str,
                                 dest_name: str,
                                 e_clean: Dict[str, torch.Tensor],
                                 e_corr: Dict[str, torch.Tensor],
                                 grad_dest: torch.Tensor) -> float:
        """Calculate edge score using VJP"""
        dest = e_clean.get(dest_name)
        if not isinstance(dest, torch.Tensor) or not isinstance(grad_dest, torch.Tensor):
            return 0.0

        src = e_clean.get(src_name)
        if src is None:
            return 0.0

        # Δe_src
        e_corr_src = e_corr.get(src_name)
        if e_corr_src is None:
            return 0.0
        src_aln, e_corr_src_aln = self._right_align_tuples(src, e_corr_src)
        delta_src = e_corr_src_aln - src_aln

        # Directly compute vjp for src
        vjp = torch.autograd.grad(
            outputs=dest, inputs=src, grad_outputs=grad_dest,
            retain_graph=True, allow_unused=True
        )[0]

        # If src is head view and returns None, compute entire a{l} block, then slice
        if vjp is None and src_name.startswith('a') and '.h' in src_name:
            base = src_name.split('.')[0]      # e.g., a31
            try:
                h = int(src_name.split('.h')[1])
            except Exception:
                h = 0
            src_full = e_clean.get(base)
            if isinstance(src_full, torch.Tensor):
                vjp_full = torch.autograd.grad(
                    outputs=dest, inputs=src_full, grad_outputs=grad_dest,
                    retain_graph=True, allow_unused=True
                )[0]
                if isinstance(vjp_full, torch.Tensor):
                    vjp = self._slice_head(vjp_full, h)

        if vjp is None:
            return 0.0

        vjp_aln, delta_src_aln = self._right_align_tuples(vjp, delta_src)
        return torch.sum(vjp_aln * delta_src_aln).item()

    def analyze(self, clean_questions: List[str], corrupted_prompts: List[str]) -> Tuple[Dict[Tuple[str, str], float], List[Dict[str, str]]]:
        """
        Execute EAP analysis
        
        Args:
            clean_questions: List of clean questions
            corrupted_prompts: List of corrupted prompts
            
        Returns:
            (Edge scores dictionary, Generation results list)
        """
        edge_scores = {edge: 0.0 for edge in self.edges}
        generation_results = []
        gen_config = self.config.get("generation_config", {})
        num_valid_samples = 0

        print(f"[Analysis] Total samples={len(clean_questions)}, layers={self.config['layers_to_analyze']}, mode={self.edge_score_mode}")

        for i in tqdm(range(len(clean_questions)), desc="EAP Analysis"):
            clean_prompt = clean_questions[i]
            corrupted_prompt = corrupted_prompts[i]
            print(f"\n[Sample {i}] clean_prompt_len={len(clean_prompt)}, corrupted_prompt_len={len(corrupted_prompt)}")

            try:
                # Format prompts
                clean_messages = [{"role": "user", "content": clean_prompt}]
                corrupted_messages = [{"role": "user", "content": corrupted_prompt}]

                clean_prompt_formatted = self.tokenizer.apply_chat_template(
                    clean_messages, tokenize=False, add_generation_prompt=True)
                corrupted_prompt_formatted = self.tokenizer.apply_chat_template(
                    corrupted_messages, tokenize=False, add_generation_prompt=True)

                inputs_clean = self.tokenizer(clean_prompt_formatted, return_tensors="pt").to(self.device)
                inputs_corrupted = self.tokenizer(corrupted_prompt_formatted, return_tensors="pt").to(self.device)
                print(f"[Sample {i}] Tokenization: clean.input_ids.shape={tuple(inputs_clean['input_ids'].shape)}, "
                      f"corrupt.input_ids.shape={tuple(inputs_corrupted['input_ids'].shape)}")

                # Generation (record only)
                with torch.no_grad():
                    gen_kwargs = gen_config.copy()
                    if self.tokenizer.pad_token_id is not None:
                        gen_kwargs['pad_token_id'] = self.tokenizer.eos_token_id
                    outputs_clean_gen = self.model.generate(**inputs_clean, **gen_kwargs)
                    answer_clean = self.tokenizer.decode(outputs_clean_gen[0], skip_special_tokens=True)
                    outputs_corrupted_gen = self.model.generate(**inputs_corrupted, **gen_kwargs)
                    answer_corrupted = self.tokenizer.decode(outputs_corrupted_gen[0], skip_special_tokens=True)
                generation_results.append({
                    "clean_prompt_formatted": clean_prompt_formatted, "clean_answer": answer_clean,
                    "corrupted_prompt_formatted": corrupted_prompt_formatted, "corrupted_answer": answer_corrupted
                })

                # 1) Corrupted: Get activations (no gradients)
                self.model.zero_grad()
                with torch.no_grad():
                    self._register_forward_hooks()
                    self.model(**inputs_corrupted)
                    e_corr_dict = {k: v for k, v in self.activations.items()}

                # 2) Clean: Get activations + calculate metric + gradients
                self.model.zero_grad()
                ckpt_was_on = bool(getattr(self.model, "is_gradient_checkpointing", False))
                if self.temp_disable_ckpt_for_grad and ckpt_was_on:
                    print("[Checkpoint] Temporarily disabling gradient checkpointing")
                    try:
                        self.model.gradient_checkpointing_disable()
                    except Exception as e:
                        print(f"[Checkpoint] Disable failed: {e}")

                self._register_forward_hooks()
                clean_outputs = self.model(**inputs_clean)
                e_clean_dict = self.activations.copy()
                metric_L = self._calculate_metric(clean_outputs.logits, clean_prompt)
                print(f"[Sample {i}] Metric L={metric_L.item():.6f}")
                
                if not torch.isfinite(metric_L) or metric_L.item() == 0.0:
                    print(f"[Sample {i}] Invalid metric, skipping")
                    if self.temp_disable_ckpt_for_grad and ckpt_was_on:
                        try:
                            self.model.gradient_checkpointing_enable()
                            print("[Checkpoint] Re-enabling gradient checkpointing")
                        except Exception as e:
                            print(f"[Checkpoint] Enable failed: {e}")
                    continue

                num_valid_samples += 1

                # Calculate gradients
                activations_to_grad, node_names_for_grad = [], []

                # Collect all activations that need gradient calculation
                for name, act in e_clean_dict.items():
                    if name in self.nodes and isinstance(act, torch.Tensor) and act.requires_grad:
                        if name.startswith('a') and ('.h' in name):
                            continue  # Skip head views to avoid None
                        activations_to_grad.append(act)
                        node_names_for_grad.append(name)

                gradients = torch.autograd.grad(
                    outputs=metric_L, inputs=activations_to_grad, allow_unused=True, retain_graph=True
                )
                grad_dict: Dict[str, torch.Tensor] = {}
                for nm, g in zip(node_names_for_grad, gradients):
                    if g is not None:
                        grad_dict[nm] = g

                # Use ∂L/∂a{l} slicing to fill ∂L/∂a{l}.h{h}
                for l in self.config['layers_to_analyze']:
                    g_full = grad_dict.get(f"a{l}")
                    if isinstance(g_full, torch.Tensor):
                        for h in range(self.num_heads):
                            name_h = f"a{l}.h{h}"
                            if not isinstance(grad_dict.get(name_h), torch.Tensor):
                                grad_dict[name_h] = self._slice_head(g_full, h)

                if self.temp_disable_ckpt_for_grad and ckpt_was_on:
                    try:
                        self.model.gradient_checkpointing_enable()
                        print("[Checkpoint] Re-enabling gradient checkpointing")
                    except Exception as e:
                        print(f"[Checkpoint] Enable failed: {e}")

                # 3) Calculate edge scores using VJP mode
                count_used = 0
                for src_node, dest_node in self.edges:
                    grad_dest = grad_dict.get(dest_node)
                    if grad_dest is None:
                        continue
                    s = self._edge_score_vjp_fallback(src_node, dest_node, e_clean_dict, e_corr_dict, grad_dest)
                    edge_scores[(src_node, dest_node)] += s
                    count_used += 1
                print(f"[Sample {i}] VJP mode edges used={count_used}")

            finally:
                self._clear_hooks_and_data()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if num_valid_samples > 0:
            for edge in edge_scores:
                edge_scores[edge] /= num_valid_samples
        print(f"[Analysis] Completed. Valid samples={num_valid_samples}")
        return edge_scores, generation_results
