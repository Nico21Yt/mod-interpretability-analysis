#!/usr/bin/env python3
"""
Activation Data Preparer
Used for extracting and analyzing activation data from model components
"""

import json
import os
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import defaultdict


class ActivationPreparer:
    """Activation Data Preparer"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config["model_name"]
        
        # Handle device configuration
        device_config = config.get("device", "auto")
        if device_config == "auto":
            # Auto-detect: if CUDA is available and device count > 0, use cuda:0
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            # Manually specified device
            self.device = device_config
        
        if config["output_config"]["verbose"]:
            print(f"Loading model: {self.model_name}")
            print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
            device_map={"": self.device}
        )
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract_numbers_from_prompt(self, prompt: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract numbers from prompt"""
        pattern = r'\((\d+)\s*\+\s*(\d+)\)\s*(?:mod|modulo)\s*(\d+)'
        match = re.search(pattern, prompt)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return None, None, None
    
    def get_component_activations(self, prompt: str, component_id: str) -> Optional[torch.Tensor]:
        """Get activation data for specified component"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        activations = {}
        hook_handle = None

        def make_hook(name, head_idx=None):
            def hook(module, input, output):
                hidden_states = output[0].detach()
                if head_idx is not None:
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    num_heads = self.model.config.num_attention_heads
                    head_dim = hidden_dim // num_heads
                    reshaped_states = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
                    head_activation = reshaped_states[:, :, head_idx, :].clone()
                    activations[name] = head_activation
                else:
                    activations[name] = hidden_states.clone()
            return hook

        try:
            if component_id.startswith('a'):
                layer_match = re.match(r'a(\d+)\.h(\d+)', component_id)
                if layer_match:
                    layer_idx, head_idx = map(int, layer_match.groups())
                    target_module = self.model.model.layers[layer_idx].self_attn
                    hook_handle = target_module.register_forward_hook(make_hook(component_id, head_idx=head_idx))
            elif component_id.startswith('m'):
                layer_match = re.match(r'm(\d+)', component_id)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    target_module = self.model.model.layers[layer_idx].mlp
                    hook_handle = target_module.register_forward_hook(make_hook(component_id))
            
            if hook_handle is None:
                print(f"Warning: Unable to register hook for component {component_id}")
                return None
                
            with torch.no_grad():
                self.model(**inputs)
        finally:
            if hook_handle:
                hook_handle.remove()
                
        return activations.get(component_id, None)

    def compute_token_importance_scores(self, activations: torch.Tensor) -> List[float]:
        """Calculate token importance scores"""
        if activations is None: 
            return []
        if len(activations.shape) == 3: 
            activations = activations[0]
        token_importance = torch.norm(activations.float(), dim=-1)
        importance_scores = token_importance.cpu().numpy().tolist()
        
        if self.config["activation_analysis"]["normalize_scores"] and importance_scores:
            max_score = max(importance_scores)
            if max_score > 1e-6:
                importance_scores = [score / max_score for score in importance_scores]
        return importance_scores
    
    def prepare_single_sample_data(self, prompt: str, component_id: str) -> Optional[Dict[str, Any]]:
        """Prepare activation data for single sample"""
        a, b, c = self.extract_numbers_from_prompt(prompt)
        correct_answer = (a + b) % c if all(x is not None for x in [a, b, c]) else None
        activations = self.get_component_activations(prompt, component_id)
        
        if activations is None: 
            return None
            
        importance_scores = self.compute_token_importance_scores(activations)
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))
        min_len = min(len(tokens), len(importance_scores))
        tokens, importance_scores = tokens[:min_len], importance_scores[:min_len]
        
        token_importance_pairs = [
            {"token": t, "importance_score": round(s, 4)} 
            for t, s in zip(tokens, importance_scores)
        ]
        
        return {
            "prompt": prompt, 
            "component_id": component_id,
            "tokens_with_importance": token_importance_pairs,
            "task_info": {
                "numbers": [a, b, c] if a is not None else None, 
                "correct_answer": correct_answer, 
                "prompt_type": "modular_arithmetic"
            },
            "activation_stats": {
                "num_tokens": len(tokens), 
                "max_importance": max(importance_scores, default=0), 
                "mean_importance": np.mean(importance_scores).item() if importance_scores else 0, 
                "activation_shape": list(activations.shape)
            }
        }
    
    def prepare_batch_data(self, prompts: List[str], component_ids: List[str], output_dir: str = None) -> Dict[str, Any]:
        """Prepare batch activation data"""
        if output_dir is None: 
            output_dir = self.config.get("output_dir", "activation_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = {
            "metadata": {
                "model_name": self.model_name, 
                "task": "modular_arithmetic", 
                "total_prompts": len(prompts), 
                "total_components": len(component_ids)
            }, 
            "component_data": {}
        }
        
        for component_id in component_ids:
            if self.config["output_config"]["verbose"]: 
                print(f"Processing component: {component_id}")
            component_data = {"component_id": component_id, "samples": []}
            
            for i, prompt in enumerate(prompts):
                if self.config["output_config"]["verbose"]: 
                    print(f"  Processing prompt {i+1}/{len(prompts)} for {component_id}")
                sample_data = self.prepare_single_sample_data(prompt, component_id)
                if sample_data: 
                    component_data["samples"].append(sample_data)
            
            all_data["component_data"][component_id] = component_data
            
            if self.config["output_config"]["save_individual_files"]:
                with open(os.path.join(output_dir, f"{component_id}_activation_data.json"), 'w', encoding='utf-8') as f: 
                    json.dump(component_data, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, "complete_activation_dataset.json"), 'w', encoding='utf-8') as f: 
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        if self.config["output_config"]["create_gpt_prompts"]: 
            self.create_gpt_ready_format(all_data, output_dir)
        
        return all_data
    
    def create_gpt_ready_format(self, data: Dict[str, Any], output_dir: str):
        """Create GPT-ready format prompts"""
        os.makedirs(output_dir, exist_ok=True)
        
        system_prompt = """你是一个专门用于大语言模型可解释性的AI助手。

项目背景和目标:
这个项目的核心研究目标是调查大语言模型在思维链(CoT)推理过程中的"不真诚"现象。
具体来说，这指的是模型生成看似逻辑的推理步骤，但其内部关键组件可能并未真正执行这些步骤。
相反，它们可能使用了"捷径"或启发式方法来直接得出答案，然后生成一个听起来合理的合理化解释。
你将分析的组件已经通过因果分析技术（如"电路追踪"）预先识别为对模型输出具有因果重要性。

你的核心任务:
你的主要功能是分析来自单个预识别组件（注意力头或MLP层）的激活数据，并在上述研究背景下推断和描述其特定功能，
特别关注任何"不真诚"或类似捷径的行为迹象。

指导原则:
你的分析必须客观，仅基于提供的数据。

在你的分析中，密切关注可能揭示"捷径"行为的模式。例如：
- 组件是否真正参与逐步计算，还是只关注与最终答案相关的token？
- 它是否忽略中间数值？
- 它是否对某些非计算关键词（如"逐步"）过度敏感？

输出格式要求:
你的回复必须是一个单一的、有效的JSON对象，严格遵循指定的输出模式。不要包含任何介绍性文本、
markdown格式（如```json）、摘要或JSON对象之外的解释。"""
        
        with open(os.path.join(output_dir, "system_prompt.txt"), 'w', encoding='utf-8') as f:
            f.write(system_prompt)

        for component_id, component_data in data["component_data"].items():
            # 准备数据块
            match = re.search(r'([am])(\d+)', component_id)
            component_type = "注意力头" if match.group(1) == 'a' else "MLP"
            layer_index = int(match.group(2))

            # 创建用户prompt
            user_prompt = f"""分析以下模型组件的功能，基于提供的激活数据。

### 组件元数据 ###
组件ID: {component_id}
组件类型: {component_type}
层索引: {layer_index}

### 激活数据 (每个样本的前{self.config["activation_analysis"]["top_tokens_display"]}个token) ###
"""
            
            samples_for_prompt = []
            top_k = self.config["activation_analysis"]["top_tokens_display"]
            for sample in component_data["samples"]:
                top_tokens = sorted(sample['tokens_with_importance'], key=lambda x: x['importance_score'], reverse=True)[:top_k]
                samples_for_prompt.append({
                    "prompt": sample["prompt"],
                    "taskInfo": sample["task_info"],
                    "topActivatedTokens": [{"token": t["token"], "score": t["importance_score"]} for t in top_tokens]
                })
            
            activation_data_block = json.dumps({"samples": samples_for_prompt}, indent=2, ensure_ascii=False)
            user_prompt += activation_data_block
            
            output_schema_str = json.dumps({
                "componentId": component_id,
                "analysis": {
                    "inferredFunction": "一个简洁的句子，总结组件的主要功能。",
                    "keyPatterns": ["基于数据观察到的具体、有证据的模式列表。"],
                    "detailedRole": "详细段落，解释此组件如何有助于解决任务。",
                    "confidence": "High、Medium或Low之一。",
                    "confidenceReasoning": "对你的置信度水平的简要说明。"
                }, 
                "metadata": {"componentType": component_type, "layerIndex": layer_index}
            }, indent=2, ensure_ascii=False)

            user_prompt += f"""

### 分析说明和输出模式 ###
基于上述数据，执行以下分析并生成严格遵循此模式的JSON输出。

1. **解释激活模式**: 查看token数据中的重复激活模式。
2. **推断功能角色**: 将激活模式综合为连贯的功能描述。
3. **生成JSON输出**: 用你的分析填充以下JSON模式。

{output_schema_str}
"""
            
            with open(os.path.join(output_dir, f"{component_id}_api_prompt.txt"), 'w', encoding='utf-8') as f:
                f.write(user_prompt)
        
        if self.config["output_config"]["verbose"]:
            print(f"API prompts保存到: {output_dir}/")
