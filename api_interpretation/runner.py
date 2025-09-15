#!/usr/bin/env python3
"""
API Interpretation Runner
Responsible for executing the complete API call workflow
"""

import os
import json
from typing import Dict, Any, List

from .api_client import APIClient


class APIIterpretationRunner:
    """API Interpretation Runner"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def find_analysis_run_directory(self, run_id: str = None) -> str:
        """Find analysis run directory"""
        analysis_runs_dir = self.config.get("analysis_runs_dir", "analysis_runs")
        
        if run_id:
            run_dir = os.path.join(analysis_runs_dir, run_id)
            if not os.path.exists(run_dir):
                raise FileNotFoundError(f"Specified run directory not found: '{run_dir}'")
            print(f"Using specified analysis run: {run_id}")
            return run_dir
        else:
            # Auto-detect latest run
            if not os.path.exists(analysis_runs_dir):
                raise FileNotFoundError(f"Analysis runs directory not found: '{analysis_runs_dir}'")
            
            all_runs = [d for d in os.listdir(analysis_runs_dir) 
                       if os.path.isdir(os.path.join(analysis_runs_dir, d))]
            if not all_runs:
                raise FileNotFoundError(f"No analysis runs found in '{analysis_runs_dir}'")
            
            latest_run = max(all_runs, key=lambda d: os.path.getmtime(os.path.join(analysis_runs_dir, d)))
            run_dir = os.path.join(analysis_runs_dir, latest_run)
            print(f"Auto-detected latest analysis run: {latest_run}")
            return run_dir
    
    def load_prompts_data(self, run_dir: str) -> Dict[str, Dict[str, str]]:
        """Load prompts data"""
        prompts_dir = os.path.join(run_dir, "02_prompts_for_api")
        responses_dir = os.path.join(run_dir, "03_api_responses")
        
        if not os.path.exists(prompts_dir):
            raise FileNotFoundError(f"Prompts directory not found: '{prompts_dir}'. Please run activation analysis module first.")
        
        # Create responses directory
        os.makedirs(responses_dir, exist_ok=True)
        
        # Load system prompt
        system_prompt_path = os.path.join(prompts_dir, "system_prompt.txt")
        if not os.path.exists(system_prompt_path):
            raise FileNotFoundError("'system_prompt.txt' not found. Cannot continue.")
        
        with open(system_prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        # Load component prompts
        prompt_files = sorted([f for f in os.listdir(prompts_dir) if f.endswith("_api_prompt.txt")])
        if not prompt_files:
            raise ValueError("No API prompt files found for processing.")
        
        prompts_data = {}
        for filename in prompt_files:
            component_id = filename.replace("_api_prompt.txt", "")
            
            # Check if response already exists
            response_path = os.path.join(responses_dir, f"{component_id}_response.json")
            if os.path.exists(response_path):
                print(f"  [Skip] Response file for component {component_id} already exists")
                continue
            
            # Load user prompt
            user_prompt_path = os.path.join(prompts_dir, filename)
            with open(user_prompt_path, 'r', encoding='utf-8') as f:
                user_prompt = f.read()
            
            prompts_data[component_id] = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt
            }
        
        return prompts_data
    
    def save_responses(self, responses: Dict[str, Dict[str, Any]], run_dir: str):
        """Save API responses"""
        responses_dir = os.path.join(run_dir, "03_api_responses")
        
        for component_id, response in responses.items():
            response_path = os.path.join(responses_dir, f"{component_id}_response.json")
            with open(response_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False)
            print(f"  [Save] Successfully saved analysis to '{response_path}'")
    
    def generate_summary_report(self, responses: Dict[str, Dict[str, Any]], run_dir: str):
        """Generate summary report"""
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
                    "inferred_function": analysis.get("inferredFunction", "Not provided"),
                    "confidence": analysis.get("confidence", "Unknown"),
                    "key_patterns": analysis.get("keyPatterns", []),
                    "detailed_role": analysis.get("detailedRole", "Not provided")
                }
            else:
                summary_data["component_analyses"][component_id] = {
                    "error": response["error"],
                    "status": "failed"
                }
        
        # Save summary report
        summary_path = os.path.join(run_dir, "04_summary_report.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary report saved to: {summary_path}")
        
        # Print brief statistics
        print(f"\n=== Analysis Completion Statistics ===")
        print(f"Total components: {summary_data['metadata']['total_components']}")
        print(f"Successful analyses: {summary_data['metadata']['successful_analyses']}")
        print(f"Failed analyses: {summary_data['metadata']['failed_analyses']}")
        
        return summary_data
    
    def run_interpretation(self, run_id: str = None) -> str:
        """
        Run complete API interpretation workflow
        
        Args:
            run_id: Analysis run ID to process, None means auto-detect latest
            
        Returns:
            Analysis run directory path
        """
        print("=== Starting API Component Interpretation ===")
        print("=" * 60)
        
        try:
            # 1. Find analysis run directory
            run_id = run_id or self.config.get("run_id_to_process")
            run_dir = self.find_analysis_run_directory(run_id)
            
            # 2. Define I/O directories
            prompts_dir = os.path.join(run_dir, "02_prompts_for_api")
            responses_dir = os.path.join(run_dir, "03_api_responses")
            os.makedirs(responses_dir, exist_ok=True)
            
            print(f"Using analysis run directory: {run_dir}")
            print(f"Prompts directory: {prompts_dir}")
            print(f"Responses directory: {responses_dir}")
            
            # 3. Load prompts data
            prompts_data = self.load_prompts_data(run_dir)
            
            if not prompts_data:
                print("[Info] No new API prompt files need processing.")
                return run_dir
            
            # 4. Initialize API client
            print("\nInitializing API client...")
            api_client = APIClient(self.config)
            
            # 5. Batch API calls
            print(f"\nStarting to process {len(prompts_data)} components...")
            delay = self.config.get("delay_between_calls", 1)
            responses = api_client.batch_call_api(prompts_data, delay)
            
            # 6. Save responses
            print(f"\nSaving API responses...")
            self.save_responses(responses, run_dir)
            
            # 7. Generate summary report
            print(f"\nGenerating summary report...")
            summary_data = self.generate_summary_report(responses, run_dir)
            
            print("\n=== API Interpretation Completed ===")
            print(f"Results saved in: {run_dir}")
            print(f"\nNext steps:")
            print(f"  1. Check '{os.path.join(run_dir, '03_api_responses')}/' directory for detailed analysis results")
            print(f"  2. Check '{os.path.join(run_dir, '04_summary_report.json')}' for summary report")
            print(f"  3. Conduct further research or validation based on analysis results")
            
            return run_dir
            
        except Exception as e:
            print(f"\nUnexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            raise
