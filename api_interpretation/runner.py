#!/usr/bin/env python3
"""
API解释运行器
负责执行API调用的完整流程
"""

import os
import json
from typing import Dict, Any, List

from .api_client import APIClient


class APIIterpretationRunner:
    """API解释运行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def find_analysis_run_directory(self, run_id: str = None) -> str:
        """查找分析运行目录"""
        analysis_runs_dir = self.config.get("analysis_runs_dir", "analysis_runs")
        
        if run_id:
            run_dir = os.path.join(analysis_runs_dir, run_id)
            if not os.path.exists(run_dir):
                raise FileNotFoundError(f"指定的运行目录未找到: '{run_dir}'")
            print(f"使用指定的分析运行: {run_id}")
            return run_dir
        else:
            # 自动检测最新的运行
            if not os.path.exists(analysis_runs_dir):
                raise FileNotFoundError(f"分析运行目录未找到: '{analysis_runs_dir}'")
            
            all_runs = [d for d in os.listdir(analysis_runs_dir) 
                       if os.path.isdir(os.path.join(analysis_runs_dir, d))]
            if not all_runs:
                raise FileNotFoundError(f"在 '{analysis_runs_dir}' 中未找到分析运行")
            
            latest_run = max(all_runs, key=lambda d: os.path.getmtime(os.path.join(analysis_runs_dir, d)))
            run_dir = os.path.join(analysis_runs_dir, latest_run)
            print(f"自动检测到最新分析运行: {latest_run}")
            return run_dir
    
    def load_prompts_data(self, run_dir: str) -> Dict[str, Dict[str, str]]:
        """加载prompts数据"""
        prompts_dir = os.path.join(run_dir, "02_prompts_for_api")
        responses_dir = os.path.join(run_dir, "03_api_responses")
        
        if not os.path.exists(prompts_dir):
            raise FileNotFoundError(f"Prompts目录未找到: '{prompts_dir}'。请先运行激活分析模块。")
        
        # 创建响应目录
        os.makedirs(responses_dir, exist_ok=True)
        
        # 加载系统prompt
        system_prompt_path = os.path.join(prompts_dir, "system_prompt.txt")
        if not os.path.exists(system_prompt_path):
            raise FileNotFoundError("'system_prompt.txt' 未找到。无法继续。")
        
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        # 加载组件prompts
        prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith("_api_prompt.txt")])
        if not prompt_files:
            raise ValueError("未找到API prompt文件进行处理。")
        
        prompts_data = {}
        for filename in prompt_files:
            component_id = filename.replace("_api_prompt.txt", "")
            
            # 检查是否已经有响应
            response_path = os.path.join(responses_dir, f"{component_id}_response.json")
            if os.path.exists(response_path):
                print(f"  [跳过] 组件 {component_id} 的响应文件已存在")
                continue
            
            # 加载用户prompt
            user_prompt_path = os.path.join(prompts_dir, filename)
            with open(user_prompt_path, 'r', encoding='utf-8') as f:
                user_prompt = f.read()
            
            prompts_data[component_id] = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }
        
        return prompts_data
    
    def save_responses(self, responses: Dict[str, Dict[str, Any]], run_dir: str):
        """保存API响应"""
        responses_dir = os.path.join(run_dir, "03_api_responses")
        
        for component_id, response in responses.items():
            response_path = os.path.join(responses_dir, f"{component_id}_response.json")
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            print(f"  [保存] 成功保存分析到 '{response_path}'")
    
    def generate_summary_report(self, responses: Dict[str, Dict[str, Any]], run_dir: str):
        """生成汇总报告"""
        summary_data = {
            "metadata": {
                "total_components": len(responses),
                "successful_analyses": sum(1 for r in responses.values() if "error" not in r),
                "failed_analyses": sum(1 for r in responses.values() if "error" in r),
            },
            "component_analyses": {}
        }
        
        for component_id, response in responses.items():
            if "error" not in response:
                analysis = response.get("analysis", {})
                summary_data["component_analyses"][component_id] = {
                    "inferred_function": analysis.get("inferredFunction", "未提供"),
                    "confidence": analysis.get("confidence", "未知"),
                    "key_patterns": analysis.get("keyPatterns", []),
                    "detailed_role": analysis.get("detailedRole", "未提供")
                }
            else:
                summary_data["component_analyses"][component_id] = {
                    "error": response["error"],
                    "status": "failed"
                }
        
        # 保存汇总报告
        summary_path = os.path.join(run_dir, "04_summary_report.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n汇总报告保存到: {summary_path}")
        
        # 打印简要统计
        print(f"\n=== 分析完成统计 ===")
        print(f"总组件数: {summary_data['metadata']['total_components']}")
        print(f"成功分析: {summary_data['metadata']['successful_analyses']}")
        print(f"失败分析: {summary_data['metadata']['failed_analyses']}")
        
        return summary_data
    
    def run_interpretation(self, run_id: str = None) -> str:
        """
        运行完整的API解释流程
        
        Args:
            run_id: 要处理的分析运行ID，None表示自动检测最新的
            
        Returns:
            分析运行目录路径
        """
        print("=== 开始API组件解释 ===")
        print("=" * 60)
        
        try:
            # 1. 查找分析运行目录
            run_id = run_id or self.config.get("run_id_to_process")
            run_dir = self.find_analysis_run_directory(run_id)
            
            # 2. 定义I/O目录
            prompts_dir = os.path.join(run_dir, "02_prompts_for_api")
            responses_dir = os.path.join(run_dir, "03_api_responses")
            os.makedirs(responses_dir, exist_ok=True)
            
            print(f"使用分析运行目录: {run_dir}")
            print(f"Prompts目录: {prompts_dir}")
            print(f"响应目录: {responses_dir}")
            
            # 3. 加载prompts数据
            prompts_data = self.load_prompts_data(run_dir)
            
            if not prompts_data:
                print("[信息] 没有新的API prompt文件需要处理。")
                return run_dir
            
            # 4. 初始化API客户端
            print("\n初始化API客户端...")
            api_client = APIClient(self.config)
            
            # 5. 批量调用API
            print(f"\n开始处理 {len(prompts_data)} 个组件...")
            delay = self.config.get("delay_between_calls", 1)
            responses = api_client.batch_call_api(prompts_data, delay)
            
            # 6. 保存响应
            print(f"\n保存API响应...")
            self.save_responses(responses, run_dir)
            
            # 7. 生成汇总报告
            print(f"\n生成汇总报告...")
            summary_data = self.generate_summary_report(responses, run_dir)
            
            print("\n=== API解释完成 ===")
            print(f"结果保存在: {run_dir}")
            print(f"\n下一步:")
            print(f"  1. 查看 '{os.path.join(run_dir, '03_api_responses')}/' 目录中的详细分析结果")
            print(f"  2. 查看 '{os.path.join(run_dir, '04_summary_report.json')}' 获取汇总报告")
            print(f"  3. 根据分析结果进行进一步的研究或验证")
            
            return run_dir
            
        except Exception as e:
            print(f"\n发生意外错误: {e}")
            import traceback
            traceback.print_exc()
            raise
