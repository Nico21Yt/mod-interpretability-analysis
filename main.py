#!/usr/bin/env python3
"""
Main execution script
Integrates three modules in complete workflow: Circuit Tracing -> Activation Analysis -> API Interpretation
Supports execution flow control through configuration files
"""

import os
import sys
import argparse
from typing import Dict, Any, List

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import config manager first to set environment variables
from config_manager import get_config_manager, reload_config

# Set CUDA environment variables (before importing torch-related modules)
config_manager = get_config_manager()
base_config = config_manager.get_base_config()
gpu_id = base_config.get("gpu_id")
if gpu_id is not None and gpu_id != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")

# Now safely import other modules
from circuit_tracing import CircuitTracingRunner
from activation_analysis import ActivationAnalysisRunner
from api_interpretation import APIIterpretationRunner


class MainRunner:
    """Main runner"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_manager = get_config_manager(config_file)
        self.circuit_runner = None
        self.activation_runner = None
        self.api_runner = None
        
    def run_circuit_tracing(self, config_updates: Dict[str, Any] = None) -> str:
        """Run circuit tracing module"""
        print("\n" + "="*80)
        print("Step 1/3: Circuit Tracing (EAP Analysis)")
        print("="*80)
        
        config = self.config_manager.get_circuit_tracing_config()
        if config_updates:
            self.config_manager.update_config("circuit_tracing", config_updates)
            config = self.config_manager.get_circuit_tracing_config()
        
        self.circuit_runner = CircuitTracingRunner(config)
        prompts_file = self.config_manager.get_prompts_file_path()
        results_dir = self.circuit_runner.run_analysis(prompts_file)
        
        if results_dir:
            print(f"\nCircuit tracing completed! Results saved in: {results_dir}")
            return results_dir
        else:
            raise RuntimeError("Circuit tracing failed")
    
    def run_activation_analysis(self, results_index: int = None, config_updates: Dict[str, Any] = None) -> str:
        """Run activation analysis module"""
        print("\n" + "="*80)
        print("Step 2/3: Activation Analysis")
        print("="*80)
        
        config = self.config_manager.get_activation_analysis_config()
        if config_updates:
            self.config_manager.update_config("activation_analysis", config_updates)
            config = self.config_manager.get_activation_analysis_config()
        
        self.activation_runner = ActivationAnalysisRunner(config)
        analysis_dir = self.activation_runner.run_analysis(results_index)
        
        print(f"\nActivation analysis completed! Results saved in: {analysis_dir}")
        return analysis_dir
    
    def run_api_interpretation(self, run_id: str = None, config_updates: Dict[str, Any] = None) -> str:
        """Run API interpretation module"""
        print("\n" + "="*80)
        print("Step 3/3: API Interpretation")
        print("="*80)
        
        config = self.config_manager.get_api_interpretation_config()
        if config_updates:
            self.config_manager.update_config("api_interpretation", config_updates)
            config = self.config_manager.get_api_interpretation_config()
        
        self.api_runner = APIIterpretationRunner(config)
        final_dir = self.api_runner.run_interpretation(run_id)
        
        print(f"\nAPI interpretation completed! Results saved in: {final_dir}")
        return final_dir
    
    def run_full_pipeline(self, 
                         circuit_config: Dict[str, Any] = None,
                         activation_config: Dict[str, Any] = None,
                         api_config: Dict[str, Any] = None) -> str:
        """Run complete analysis pipeline"""
        print("Starting model interpretability analysis pipeline")
        print("="*80)
        
        # Display configuration summary
        if self.config_manager.is_verbose():
            self.config_manager.print_config_summary()
        
        # Pre-check
        if self.config_manager.should_pre_check():
            print("\nPerforming pre-check...")
            errors = self.config_manager.validate_config()
            if errors:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                raise ValueError("Configuration validation failed")
            print("Pre-check passed")
        
        # Get tasks to execute
        tasks = self.config_manager.get_tasks_to_execute()
        print(f"\nExecuting tasks: {', '.join(tasks)}")
        
        try:
            results = {}
            
            # Execute tasks
            for i, task in enumerate(tasks, 1):
                print(f"\n{'='*80}")
                print(f"Step {i}/{len(tasks)}: {task}")
                print("="*80)
                
                if task == "circuit_tracing":
                    results['circuit_tracing'] = self.run_circuit_tracing(circuit_config)
                elif task == "activation_analysis":
                    # If circuit tracing was done before, use its results
                    if 'circuit_tracing' in results:
                        results_index = int(os.path.basename(results['circuit_tracing']))
                    else:
                        results_index = None
                    results['activation_analysis'] = self.run_activation_analysis(results_index, activation_config)
                elif task == "api_interpretation":
                    # If activation analysis was done before, use its results
                    if 'activation_analysis' in results:
                        run_id = os.path.basename(results['activation_analysis'])
                    else:
                        run_id = None
                    results['api_interpretation'] = self.run_api_interpretation(run_id, api_config)
            
            # Display final results
            print("\n" + "="*80)
            print("Analysis pipeline completed!")
            print("="*80)
            
            for task, result_dir in results.items():
                print(f"{task} results: {result_dir}")
            
            # Return the result of the last task
            final_dir = list(results.values())[-1] if results else None
            
            if final_dir:
                print(f"\nMain result files:")
                if 'circuit_tracing' in results:
                    circuit_dir = results['circuit_tracing']
                    print(f"  - Circuit graph: {os.path.join(circuit_dir, 'pruned_graph_info.json')}")
                if 'activation_analysis' in results:
                    activation_dir = results['activation_analysis']
                    print(f"  - Activation data: {os.path.join(activation_dir, '01_prepared_activations/')}")
                if 'api_interpretation' in results:
                    api_dir = results['api_interpretation']
                    print(f"  - API responses: {os.path.join(api_dir, '03_api_responses/')}")
                    print(f"  - Summary report: {os.path.join(api_dir, '04_summary_report.json')}")
            
            return final_dir
            
        except Exception as e:
            print(f"\nPipeline execution failed: {e}")
            if self.config_manager.should_stop_on_error():
                import traceback
                traceback.print_exc()
                raise
            else:
                print("Continuing with other tasks...")
                return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Model interpretability analysis pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Configuration file path (default: config.yaml)")
    parser.add_argument("--mode", choices=["full", "circuit", "activation", "api"], 
                       default="full", help="Run mode (default: full)")
    parser.add_argument("--results-index", type=int, help="Specify result index to analyze")
    parser.add_argument("--run-id", type=str, help="Specify analysis run ID to process")
    parser.add_argument("--gpu-id", type=str, help="Specify GPU ID")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode")
    parser.add_argument("--num-samples", type=int, help="Number of samples")
    parser.add_argument("--top-components", type=int, help="Number of components to analyze")
    parser.add_argument("--no-pre-check", action="store_true", help="Skip pre-check")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--show-config", action="store_true", help="Show configuration summary and exit")
    
    args = parser.parse_args()
    
    try:
        # Create main runner
        runner = MainRunner(args.config)
        
        # Show configuration summary
        if args.show_config:
            runner.config_manager.print_config_summary()
            return 0
        
        # Prepare configuration updates
        circuit_config_updates = {}
        activation_config_updates = {}
        api_config_updates = {}
        execution_config_updates = {}
        
        if args.gpu_id:
            circuit_config_updates["gpu_id"] = args.gpu_id
        if args.num_samples:
            circuit_config_updates["num_samples"] = args.num_samples
        if args.top_components:
            activation_config_updates["top_components"] = args.top_components
        if args.api_key:
            api_config_updates["api_key"] = args.api_key
        if args.test_mode:
            activation_config_updates["test_mode_config"] = {"enabled": True}
        if args.no_pre_check:
            execution_config_updates["pre_check"] = False
        if args.verbose:
            execution_config_updates["verbose"] = True
        
        # Apply configuration updates
        if circuit_config_updates:
            runner.config_manager.update_config("circuit_tracing", circuit_config_updates)
        if activation_config_updates:
            runner.config_manager.update_config("activation_analysis", activation_config_updates)
        if api_config_updates:
            runner.config_manager.update_config("api_interpretation", api_config_updates)
        if execution_config_updates:
            runner.config_manager.update_config("execution", execution_config_updates)
        
        # Execute tasks
        if args.mode == "full":
            # Run complete pipeline
            final_dir = runner.run_full_pipeline()
            if final_dir:
                print(f"\nAnalysis completed! Check results: {final_dir}")
            else:
                print("\nAnalysis not fully completed")
            
        elif args.mode == "circuit":
            # Run circuit tracing only
            results_dir = runner.run_circuit_tracing()
            print(f"\nCircuit tracing completed! Check results: {results_dir}")
            
        elif args.mode == "activation":
            # Run activation analysis only
            analysis_dir = runner.run_activation_analysis(args.results_index)
            print(f"\nActivation analysis completed! Check results: {analysis_dir}")
            
        elif args.mode == "api":
            # Run API interpretation only
            final_dir = runner.run_api_interpretation(args.run_id)
            print(f"\nAPI interpretation completed! Check results: {final_dir}")
            
    except KeyboardInterrupt:
        print("\n\nUser interrupted program execution")
    except Exception as e:
        print(f"\n\nProgram execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
