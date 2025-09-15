#!/usr/bin/env python3
"""
EAP分析器核心类
基于原始eap_core.py重构，专注于电路追踪功能
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import re


class EAPAnalyzer:
    """
    EAP (Edge Attribution Patching) 分析器
    
    功能：
    - 捕获模型各层的激活状态
    - 计算边的重要性分数
    - 支持模块化算术任务的答案logit差异度量
    - 提供诊断功能用于调试
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device

        # 模型配置
        self.num_heads = getattr(self.model.config, 'num_attention_heads',
                                 getattr(self.model.config, 'n_head', None))
        if self.num_heads is None:
            raise ValueError("无法从模型配置中确定注意力头数量")
        self.num_layers = getattr(self.model.config, 'num_hidden_layers', None)
        print(f"[初始化] 模型={self.model.__class__.__name__}, 层数={self.num_layers}, 头数={self.num_heads}")

        # 构建计算图
        self.nodes, self.edges = self._define_graph()
        
        # 任务模式配置
        self.task_mode = config.get("task_mode", "modular_arithmetic")
        if self.task_mode == "modular_arithmetic":
            print(f"[初始化] 任务模式={self.task_mode}, 使用答案logit差异度量")
        else:
            print(f"[初始化] 任务模式={self.task_mode}, 使用默认度量")

        # 运行时状态
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        # 边分数计算模式
        self.edge_score_mode = config.get("edge_score_mode", "vjp")
        print(f"[初始化] 边分数模式={self.edge_score_mode}")

        # 诊断配置
        self.debug_layer_for_head_check = int(config.get("debug_layer_for_head_check", config['layers_to_analyze'][-1]))
        self.debug_top_heads = int(config.get("debug_top_heads", min(8, self.num_heads)))
        self.do_hook_check_once = bool(config.get("do_hook_check_once", True))
        self.do_diag_heads_once = bool(config.get("do_diag_heads_once", True))
        self.do_head_ablation_once = bool(config.get("do_head_ablation_once", False))
        print(f"[初始化] 诊断: hook_check={self.do_hook_check_once}, diag_heads={self.do_diag_heads_once}, head_ablation={self.do_head_ablation_once}")

        # 运行时控制
        self.temp_disable_ckpt_for_grad = bool(config.get("temp_disable_checkpointing_during_grad", True))
        print(f"[初始化] 临时禁用检查点={self.temp_disable_ckpt_for_grad}")

        self._printed_hook_check = False
        self._printed_diag_heads = False
        self._printed_ablation = False

    def _define_graph(self) -> Tuple[List[str], List[Tuple[str, str]]]:
        """定义计算图结构"""
        nodes, edges = [], []
        layers = self.config['layers_to_analyze']
        num_heads = self.num_heads

        # 添加节点
        nodes.append("tok_embeds")
        for l in layers:
            nodes.append(f"m{l}")
            nodes.append(f"a{l}")  # 整个a{l}块作为节点
            for h in range(num_heads):
                nodes.append(f"a{l}.h{h}")

        if not layers:
            print(f"[图] 定义了 {len(nodes)} 个节点和 {len(edges)} 条边")
            return nodes, edges

        # 添加边
        edges.append(("tok_embeds", f"m{layers[0]}"))
        for h in range(num_heads):
            edges.append(("tok_embeds", f"a{layers[0]}.h{h}"))

        for i, l_from in enumerate(layers):
            # 层内连接
            for h in range(num_heads):
                edges.append((f"a{l_from}.h{h}", f"m{l_from}"))
            edges.append((f"a{l_from}", f"m{l_from}"))

            # 层间连接
            if i + 1 < len(layers):
                l_to = layers[i + 1]
                edges.append((f"m{l_from}", f"m{l_to}"))
                edges.append((f"a{l_from}", f"m{l_to}"))
                for h_to in range(num_heads):
                    edges.append((f"a{l_from}.h{h_to}", f"m{l_to}"))
                    for h_from in range(num_heads):
                        edges.append((f"m{l_from}", f"a{l_to}.h{h_from}"))
                        edges.append((f"a{l_from}.h{h_to}", f"a{l_to}.h{h_from}"))
        
        print(f"[图] 定义了 {len(nodes)} 个节点和 {len(edges)} 条边")
        return nodes, edges

    def _extract_numbers_from_prompt(self, prompt: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """从prompt中提取数字 a, b, c"""
        match = re.search(r'\((\d+)\s*[^\d]*\s*(\d+)\)\s*[^\d]*\s*(\d+)', prompt)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None

    def _get_answer_logit(self, logits: torch.Tensor, answer: int) -> torch.Tensor:
        """获取特定答案的logit值"""
        last_token_logits = logits[:, -1, :]
        answer_str = str(answer)
        
        # 尝试多种编码方法
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
        """创建保存激活的hook"""
        def hook(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            self.activations[name] = act
        return hook

    def _save_o_proj_preact_hook(self, layer_idx: int):
        """创建保存o_proj前激活的hook"""
        num_heads = self.num_heads
        def pre_hook(module, inputs):
            x = inputs[0]  # [B, S, H], H = num_heads * head_dim
            self.activations[f"a{layer_idx}"] = x  # 整块
            bs, sl, hs = x.shape
            assert hs % num_heads == 0, f"隐藏维度 {hs} 不能被头数 {num_heads} 整除"
            hd = hs // num_heads
            x_reshaped = x.view(bs, sl, num_heads, hd)
            for h in range(num_heads):
                self.activations[f"a{layer_idx}.h{h}"] = x_reshaped[:, :, h, :]  # 仅用于取值/对齐
            return None
        return pre_hook

    def _register_forward_hooks(self):
        """注册前向传播hooks"""
        self._clear_hooks_and_data()
        
        # 嵌入层
        if hasattr(self.model, 'get_input_embeddings'):
            embed_module = self.model.get_input_embeddings()
        elif hasattr(self.model.model, 'embed_tokens'):
            embed_module = self.model.model.embed_tokens
        else:
            raise ValueError("无法在模型中找到输入嵌入模块")
        self.hooks.append(embed_module.register_forward_hook(self._save_activation_hook("tok_embeds")))
        
        # 各层
        for l in self.config['layers_to_analyze']:
            layer = self.model.model.layers[l]
            self.hooks.append(layer.mlp.register_forward_hook(self._save_activation_hook(f"m{l}")))
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                handle = layer.self_attn.o_proj.register_forward_pre_hook(self._save_o_proj_preact_hook(l))
                self.hooks.append(handle)
            else:
                raise ValueError(f"层 {l} 没有 self_attn.o_proj; 无法为头注册pre-hook")

    def _clear_hooks_and_data(self):
        """清理hooks和数据"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def _calculate_metric(self, logits: torch.Tensor, prompt: Optional[str] = None) -> torch.Tensor:
        """计算任务相关的度量"""
        if self.task_mode == "modular_arithmetic":
            if prompt is None:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            # 提取数字并计算正确答案
            a, b, c = self._extract_numbers_from_prompt(prompt)
            if a is None or b is None or c is None:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            correct_answer = (a + b) % c
            answer_logit_tensor = self._get_answer_logit(logits, correct_answer)
            return answer_logit_tensor
        else:
            # 默认使用平均logits
            return logits[:, -1, :].mean()

    def _slice_head(self, full: torch.Tensor, h: int) -> torch.Tensor:
        """从头块中切片第h个头"""
        hd = full.shape[-1] // self.num_heads
        return full[:, :, h * hd:(h + 1) * hd]

    @staticmethod
    def _right_align_tuples(*tensors: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], ...]:
        """右对齐张量元组"""
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
        """使用VJP计算边分数"""
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

        # 直接对 src 求 vjp
        vjp = torch.autograd.grad(
            outputs=dest, inputs=src, grad_outputs=grad_dest,
            retain_graph=True, allow_unused=True
        )[0]

        # 如果src是头视图且返回None，计算整个a{l}块，然后切片
        if vjp is None and src_name.startswith('a') and '.h' in src_name:
            base = src_name.split('.')[0]      # 如 a31
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
        执行EAP分析
        
        Args:
            clean_questions: 干净的问题列表
            corrupted_prompts: 损坏的prompt列表
            
        Returns:
            (边分数字典, 生成结果列表)
        """
        edge_scores = {edge: 0.0 for edge in self.edges}
        generation_results = []
        gen_config = self.config.get("generation_config", {})
        num_valid_samples = 0

        print(f"[分析] 总样本数={len(clean_questions)}, 层数={self.config['layers_to_analyze']}, 模式={self.edge_score_mode}")

        for i in tqdm(range(len(clean_questions)), desc="EAP分析"):
            clean_prompt = clean_questions[i]
            corrupted_prompt = corrupted_prompts[i]
            print(f"\n[样本 {i}] clean_prompt_len={len(clean_prompt)}, corrupted_prompt_len={len(corrupted_prompt)}")

            try:
                # 格式化prompts
                clean_messages = [{"role": "user", "content": clean_prompt}]
                corrupted_messages = [{"role": "user", "content": corrupted_prompt}]

                clean_prompt_formatted = self.tokenizer.apply_chat_template(
                    clean_messages, tokenize=False, add_generation_prompt=True)
                corrupted_prompt_formatted = self.tokenizer.apply_chat_template(
                    corrupted_messages, tokenize=False, add_generation_prompt=True)

                inputs_clean = self.tokenizer(clean_prompt_formatted, return_tensors="pt").to(self.device)
                inputs_corrupted = self.tokenizer(corrupted_prompt_formatted, return_tensors="pt").to(self.device)
                print(f"[样本 {i}] 分词: clean.input_ids.shape={tuple(inputs_clean['input_ids'].shape)}, "
                      f"corrupt.input_ids.shape={tuple(inputs_corrupted['input_ids'].shape)}")

                # 生成（仅记录）
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

                # 1) 损坏: 获取激活（无梯度）
                self.model.zero_grad()
                with torch.no_grad():
                    self._register_forward_hooks()
                    self.model(**inputs_corrupted)
                    e_corr_dict = {k: v for k, v in self.activations.items()}

                # 2) 干净: 获取激活 + 计算度量 + 梯度
                self.model.zero_grad()
                ckpt_was_on = bool(getattr(self.model, "is_gradient_checkpointing", False))
                if self.temp_disable_ckpt_for_grad and ckpt_was_on:
                    print("[检查点] 临时禁用梯度检查点")
                    try:
                        self.model.gradient_checkpointing_disable()
                    except Exception as e:
                        print(f"[检查点] 禁用失败: {e}")

                self._register_forward_hooks()
                clean_outputs = self.model(**inputs_clean)
                e_clean_dict = self.activations.copy()
                metric_L = self._calculate_metric(clean_outputs.logits, clean_prompt)
                print(f"[样本 {i}] 度量 L={metric_L.item():.6f}")
                
                if not torch.isfinite(metric_L) or metric_L.item() == 0.0:
                    print(f"[样本 {i}] 度量无效，跳过")
                    if self.temp_disable_ckpt_for_grad and ckpt_was_on:
                        try:
                            self.model.gradient_checkpointing_enable()
                            print("[检查点] 重新启用梯度检查点")
                        except Exception as e:
                            print(f"[检查点] 启用失败: {e}")
                    continue

                num_valid_samples += 1

                # 计算梯度
                activations_to_grad, node_names_for_grad = [], []

                # 收集所有需要计算梯度的激活
                for name, act in e_clean_dict.items():
                    if name in self.nodes and isinstance(act, torch.Tensor) and act.requires_grad:
                        if name.startswith('a') and ('.h' in name):
                            continue  # 跳过头视图以避免None
                        activations_to_grad.append(act)
                        node_names_for_grad.append(name)

                gradients = torch.autograd.grad(
                    outputs=metric_L, inputs=activations_to_grad, allow_unused=True, retain_graph=True
                )
                grad_dict: Dict[str, torch.Tensor] = {}
                for nm, g in zip(node_names_for_grad, gradients):
                    if g is not None:
                        grad_dict[nm] = g

                # 使用 ∂L/∂a{l} 切片来填充 ∂L/∂a{l}.h{h}
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
                        print("[检查点] 重新启用梯度检查点")
                    except Exception as e:
                        print(f"[检查点] 启用失败: {e}")

                # 3) 使用VJP模式计算边分数
                count_used = 0
                for src_node, dest_node in self.edges:
                    grad_dest = grad_dict.get(dest_node)
                    if grad_dest is None:
                        continue
                    s = self._edge_score_vjp_fallback(src_node, dest_node, e_clean_dict, e_corr_dict, grad_dest)
                    edge_scores[(src_node, dest_node)] += s
                    count_used += 1
                print(f"[样本 {i}] vjp模式边使用数={count_used}")

            finally:
                self._clear_hooks_and_data()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if num_valid_samples > 0:
            for edge in edge_scores:
                edge_scores[edge] /= num_valid_samples
        print(f"[分析] 完成。有效样本数={num_valid_samples}")
        return edge_scores, generation_results
