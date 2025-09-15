#!/usr/bin/env python3
"""
API Client
Used for calling OpenAI API for component interpretation
"""

import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI


class APIClient:
    """API Client"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = self._initialize_client()
        
    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client"""
        api_key = self.config.get("api_key")
        org_id = self.config.get("organization_id")
        
        if not api_key or api_key == "sk-...":
            raise ValueError("OpenAI API key not set in configuration")
        
        try:
            if org_id and org_id != "org-...":
                client = OpenAI(api_key=api_key, organization=org_id)
                print("[Success] OpenAI client initialized successfully (with organization ID)")
            else:
                client = OpenAI(api_key=api_key)
                print("[Success] OpenAI client initialized successfully (without organization ID)")
            return client
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client. Details: {e}")
    
    def call_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Call OpenAI API
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            
        Returns:
            API response (parsed JSON)
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
            print(f"  [Error] API error: {e}")
            return {"error": str(e)}
    
    def call_api_with_retry(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        API call with retry mechanism
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            max_retries: Maximum number of retries
            
        Returns:
            API response
        """
        for attempt in range(max_retries):
            try:
                result = self.call_api(system_prompt, user_prompt)
                if "error" not in result:
                    return result
                else:
                    print(f"  [Retry {attempt + 1}/{max_retries}] API returned error: {result.get('error')}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"  [Retry {attempt + 1}/{max_retries}] Call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return {"error": f"Failed after {max_retries} retries"}
    
    def batch_call_api(self, prompts_data: Dict[str, Dict[str, str]], 
                      delay: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Batch API calls
        
        Args:
            prompts_data: Dictionary containing component IDs and prompt data
            delay: Call interval (seconds)
            
        Returns:
            Dictionary containing all responses
        """
        results = {}
        total_components = len(prompts_data)
        
        print(f"[Info] Found {total_components} components to analyze. Starting API calls...")
        
        for i, (component_id, prompt_data) in enumerate(prompts_data.items(), 1):
            print(f"\n--- Analyzing component {i}/{total_components}: {component_id} ---")
            
            system_prompt = prompt_data.get("system_prompt", "")
            user_prompt = prompt_data.get("user_prompt", "")
            
            if not system_prompt or not user_prompt:
                print(f"  [Skip] Component {component_id} missing necessary prompt data")
                results[component_id] = {"error": "Missing prompt data"}
                continue
            
            print(f"  [API] Calling {self.config['api_model']}...")
            api_response = self.call_api_with_retry(system_prompt, user_prompt)
            results[component_id] = api_response
            
            if "error" not in api_response:
                print(f"  [Success] Successfully obtained analysis results")
            else:
                print(f"  [Failed] Unable to get valid response: {api_response.get('error')}")
            
            # Add delay to avoid API limits
            if i < total_components:
                time.sleep(delay)
        
        return results
