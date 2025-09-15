#!/usr/bin/env python3
"""
API客户端
用于调用OpenAI API进行组件解释
"""

import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI


class APIClient:
    """API客户端"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> OpenAI:
        """初始化OpenAI客户端"""
        api_key = self.config.get("api_key")
        org_id = self.config.get("organization_id")
        
        if not api_key or api_key == "sk-...":
            raise ValueError("OpenAI API密钥未在配置中设置")
        
        try:
            if org_id and org_id != "org-...":
                client = OpenAI(api_key=api_key, organization=org_id)
                print("[成功] OpenAI客户端已成功初始化（带组织ID）")
            else:
                client = OpenAI(api_key=api_key)
                print("[成功] OpenAI客户端已成功初始化（无组织ID）")
            return client
        except Exception as e:
            raise RuntimeError(f"无法初始化OpenAI客户端。详情: {e}")
    
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        调用OpenAI API
        
        Args:
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            
        Returns:
            API响应（解析后的JSON）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config["api_model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config["temperature"],
                max_tokens=self.config["max_tokens"],
                response_format={"type": "json_object"} 
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            print(f"  [错误] API错误: {e}")
            return {"error": str(e)}
    
    def call_api_with_retry(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        带重试机制的API调用
        
        Args:
            system_prompt: 系统prompt
            user_prompt: 用户prompt
            max_retries: 最大重试次数
            
        Returns:
            API响应
        """
        for attempt in range(max_retries):
            try:
                result = self.call_api(system_prompt, user_prompt)
                if "error" not in result:
                    return result
                else:
                    print(f"  [重试 {attempt + 1}/{max_retries}] API返回错误: {result.get('error')}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数退避
            except Exception as e:
                print(f"  [重试 {attempt + 1}/{max_retries}] 调用失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return {"error": f"经过 {max_retries} 次重试后仍然失败"}
    
    def batch_call_api(self, prompts_data: Dict[str, Dict[str, str]], 
                      delay: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        批量调用API
        
        Args:
            prompts_data: 包含组件ID和prompt数据的字典
            delay: 调用间隔（秒）
            
        Returns:
            包含所有响应的字典
        """
        results = {}
        total_components = len(prompts_data)
        
        print(f"[信息] 找到 {total_components} 个组件需要分析。开始API调用...")
        
        for i, (component_id, prompt_data) in enumerate(prompts_data.items(), 1):
            print(f"\n--- 分析组件 {i}/{total_components}: {component_id} ---")
            
            system_prompt = prompt_data.get("system_prompt", "")
            user_prompt = prompt_data.get("user_prompt", "")
            
            if not system_prompt or not user_prompt:
                print(f"  [跳过] 组件 {component_id} 缺少必要的prompt数据")
                results[component_id] = {"error": "缺少prompt数据"}
                continue
            
            print(f"  [API] 调用 {self.config['api_model']}...")
            api_response = self.call_api_with_retry(system_prompt, user_prompt)
            results[component_id] = api_response
            
            if "error" not in api_response:
                print(f"  [成功] 成功获取分析结果")
            else:
                print(f"  [失败] 无法获取有效响应: {api_response.get('error')}")
            
            # 添加延迟以避免API限制
            if i < total_components:
                time.sleep(delay)
        
        return results
