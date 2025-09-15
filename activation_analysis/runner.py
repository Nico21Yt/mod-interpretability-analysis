#!/usr/bin/env python3
"""
Activation Analysis Runner
Responsible for executing the complete workflow of activation data preparation and prompt generation
"""

import os
import json
import datetime
from typing import Dict, Any, List
from collections import defaultdict

from .activation_preparer import ActivationPreparer


class ActivationAnalysisRunner:
    """Activation Analysis Runner"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def find_results_directory(self, results_index: int = None) -> str:
        """Find results directory"""
        circuit_results_base = self.config.get("circuit_results_dir", "circuit_results")
        
        if results_index is not None:
            # Use specified index
            results_dir = os.path.join(circuit_results_base, str(results_index))
            if not os.path.exists(results_dir):
                raise FileNotFoundError(f"Result directory for index {results_index} not found: {results_dir}")
            print(f"Using specified result index: {results_index} ({results_dir})")
            return results_dir
        else:
            # Auto-detect latest index
            if not os.path.exists(circuit_results_base):
                raise FileNotFoundError(f"Directory '{circuit_results_base}/' not found")
            
            # Find numeric directories (new index system) and timestamp directories (old system)
            numeric_dirs = []
            timestamp_dirs = []
            for d in os.listdir(circuit_results_base):
                dir_path = os.path.join(circuit_results_base, d)
                if os.path.isdir(dir_path):
                    if d.isdigit():
                        numeric_dirs.append((int(d), d))
                    else:
                        timestamp_dirs.append(d)
            
            if numeric_dirs:
                # Use latest numeric index
                latest_index = max(numeric_dirs, key=lambda x: x[0])
                results_dir = os.path.join(circuit_results_base, latest_index[1])
                print(f"Auto-detected latest index result: {latest_index[0]} ({results_dir})")
                return results_dir
            elif timestamp_dirs:
                # Fallback to timestamp directories
                latest_dir = max(timestamp_dirs, key=lambda d: os.path.getmtime(os.path.join(circuit_results_base, d)))
                results_dir = os.path.join(circuit_results_base, latest_dir)
                print(f"Auto-detected latest timestamp result: {results_dir}")
                return results_dir
            else:
                raise FileNotFoundError(f"No result directories found in '{circuit_results_base}/''")
    
    def select_components(self, results_dir: str) -> List[str]:
        """Select components to analyze"""
        # Check test mode switch
        if self.config.get("test_mode_config", {}).get("enabled", False):
            # Test mode
            print("Running in test mode: Using manually specified components from CONFIG")
            selected_components = self.config["test_mode_config"]["components"]
            print(f"Manually selected test components:")
            for i, comp_id in enumerate(selected_components):
                print(f"  {i+1}. {comp_id}")
            return selected_components
        else:
            # Normal analysis mode
            print("Running in normal mode: Analyzing top N components from result files")
            
            pruned_graph_path = os.path.join(results_dir, "pruned_graph_info.json")
            if not os.path.exists(pruned_graph_path):
                raise FileNotFoundError(f"'pruned_graph_info.json' not found in '{results_dir}'")
            
            with open(pruned_graph_path, 'r', encoding='utf-8') as f:
                pruned_graph = json.load(f)
            
            pruned_edges = pruned_graph.get("pruned_circuit", [])
            component_scores = defaultdict(float)
            
            for edge_data in pruned_edges:
                score = abs(edge_data.get("score", 0))
                src, dst = edge_data.get("edge", " -> ").split(" -> ")
                component_scores[src] += score
                component_scores[dst] += score
            
            top_components = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)[:self.config['top_components']]
            selected_components = [comp_id for comp_id, _ in top_components]
            
            print(f"Selected top {len(selected_components)} components for analysis:")
            for i, (comp_id, score) in enumerate(top_components):
                print(f"  {i+1}. {comp_id} (importance score: {score:.3f})")
            
            return selected_components
    
    def load_prompts(self, results_dir: str) -> List[str]:
        """Load prompts"""
        generation_results_path = os.path.join(results_dir, "generation_results.json")
        if not os.path.exists(generation_results_path):
            raise FileNotFoundError(f"'generation_results.json' not found in '{results_dir}'")
        
        with open(generation_results_path, 'r', encoding='utf-8') as f:
            generation_results = json.load(f)
        
        prompts = [
            res.get("clean_prompt_formatted", "") 
            for res in generation_results 
            if res.get("clean_prompt_formatted")
        ][:self.config['max_prompts']]
        
        if not prompts:
            raise ValueError("No prompts found. Check 'generation_results.json'")
        
        print(f"Processing {len(prompts)} prompts...")
        return prompts
    
    def create_output_directory(self, results_dir: str) -> str:
        """Create output directory"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"eap_{os.path.basename(results_dir)}_{timestamp}"
        
        analysis_runs_dir = self.config.get("analysis_runs_dir", "analysis_runs")
        run_dir = os.path.join(analysis_runs_dir, run_id)
        
        dir_activations = os.path.join(run_dir, "01_prepared_activations")
        dir_prompts = os.path.join(run_dir, "02_prompts_for_api")
        os.makedirs(dir_activations, exist_ok=True)
        os.makedirs(dir_prompts, exist_ok=True)
        
        print(f"Initializing new analysis run: {run_id}")
        print(f"All outputs will be saved in: {run_dir}")
        
        return run_dir, dir_activations, dir_prompts
    
    def run_analysis(self, results_index: int = None) -> str:
        """
        Run complete activation analysis
        
        Args:
            results_index: Result index to analyze, None means auto-detect latest
            
        Returns:
            Analysis run directory path
        """
        print("=== Starting Activation Data Analysis ===")
        print("=" * 60)
        
        try:
            # 1. Find source EAP result directory
            results_index = results_index or self.config.get("results_index")
            results_dir = self.find_results_directory(results_index)
            
            # 2. Create new analysis run directory structure
            run_dir, dir_activations, dir_prompts = self.create_output_directory(results_dir)
            print("=" * 60)
            
            # 3. Select components to analyze
            selected_components = self.select_components(results_dir)
            
            if not selected_components:
                raise ValueError("No components found or selected. Check configuration or result files.")
            
            print()
            
            # 4. Load prompts
            prompts = self.load_prompts(results_dir)
            print()
            
            # 5. Initialize model and data preparer
            print("Initializing model and data preparer...")
            preparer = ActivationPreparer(self.config)
            
            # 6. Process activation data
            print("Processing activation data...")
            all_data = preparer.prepare_batch_data(prompts, selected_components, dir_activations)
            
            # 7. Create GPT-ready format
            preparer.create_gpt_ready_format(all_data, dir_prompts)
            
            print("\nActivation data preparation completed!")
            print("=" * 60)
            total_samples = sum(len(data['samples']) for data in all_data['component_data'].values())
            print(f"Total samples generated: {total_samples}")
            print(f"All outputs saved in '{run_dir}/' directory")
            print(f"\nNext steps:")
            print(f"  1. Go to '{os.path.join(run_dir, '02_prompts_for_api')}/' directory")
            print(f"  2. Use powerful LLM to process .txt file contents for functional descriptions")
            print(f"  3. Or run API interpretation module to automate this process")
            
            return run_dir
            
        except Exception as e:
            print(f"\nUnexpected error occurred: {e}")
            import traceback
            traceback.print_exc()
            raise
