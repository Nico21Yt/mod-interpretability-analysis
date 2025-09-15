#!/usr/bin/env python3
"""
Configuration Manager
Used for reading and managing YAML configuration files
"""

import os
import yaml
from typing import Dict, Any, List, Optional
import logging


class ConfigManager:
    """Configuration Manager"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Configuration file format error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file: {e}")
    
    def _setup_logging(self):
        """Setup logging"""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())
        format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[]
        )
        
        # Add console handler
        if log_config.get("console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(console_handler)
        
        # Add file handler
        log_file = log_config.get("file")
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(file_handler)
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration"""
        return self.config.get("execution", {})
    
    def get_base_config(self) -> Dict[str, Any]:
        """Get base configuration"""
        return self.config.get("base", {})
    
    def get_circuit_tracing_config(self) -> Dict[str, Any]:
        """Get circuit tracing configuration"""
        base_config = self.get_base_config()
        circuit_config = self.config.get("circuit_tracing", {})
        return {**base_config, **circuit_config}
    
    def get_activation_analysis_config(self) -> Dict[str, Any]:
        """Get activation analysis configuration"""
        base_config = self.get_base_config()
        activation_config = self.config.get("activation_analysis", {})
        return {**base_config, **activation_config}
    
    def get_api_interpretation_config(self) -> Dict[str, Any]:
        """Get API interpretation configuration"""
        base_config = self.get_base_config()
        api_config = self.config.get("api_interpretation", {})
        return {**base_config, **api_config}
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return self.config.get("data", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config.get("performance", {})
    
    def get_tasks_to_execute(self) -> List[str]:
        """Get list of tasks to execute"""
        execution_config = self.get_execution_config()
        tasks = execution_config.get("tasks", ["circuit_tracing", "activation_analysis", "api_interpretation"])
        
        # Filter out disabled tasks
        enabled_tasks = []
        for task in tasks:
            if task == "circuit_tracing" and self.config.get("circuit_tracing", {}).get("enabled", True):
                enabled_tasks.append(task)
            elif task == "activation_analysis" and self.config.get("activation_analysis", {}).get("enabled", True):
                enabled_tasks.append(task)
            elif task == "api_interpretation" and self.config.get("api_interpretation", {}).get("enabled", True):
                enabled_tasks.append(task)
        
        return enabled_tasks
    
    def is_verbose(self) -> bool:
        """Whether verbose output is enabled"""
        return self.get_execution_config().get("verbose", True)
    
    def should_stop_on_error(self) -> bool:
        """Whether to stop on error"""
        return self.get_execution_config().get("stop_on_error", True)
    
    def should_pre_check(self) -> bool:
        """Whether to perform pre-check"""
        return self.get_execution_config().get("pre_check", True)
    
    def get_prompts_file_path(self) -> str:
        """Get prompts file path"""
        data_config = self.get_data_config()
        return data_config.get("prompts_file", "data/prompts-mod-task.json")
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration (runtime)"""
        if section not in self.config:
            self.config[section] = {}
        
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.config[section] = deep_update(self.config[section], updates)
    
    def save_config(self, output_file: str = None) -> None:
        """Save configuration to file"""
        if output_file is None:
            output_file = self.config_file
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration file: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration"""
        errors = []
        
        # Check required configuration items
        if not self.config.get("base", {}).get("model_name"):
            errors.append("Missing model name configuration")
        
        # Check data files
        prompts_file = self.get_prompts_file_path()
        if not os.path.exists(prompts_file):
            errors.append(f"Data file does not exist: {prompts_file}")
        
        # Check API configuration
        api_config = self.get_api_interpretation_config()
        if api_config.get("enabled", True):
            if not api_config.get("api_key") or api_config.get("api_key") == "sk-...":
                errors.append("API interpretation module requires valid OpenAI API key")
        
        # Check task configuration
        tasks = self.get_tasks_to_execute()
        if not tasks:
            errors.append("No enabled tasks")
        
        return errors
    
    def print_config_summary(self) -> None:
        """Print configuration summary"""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        
        # Execution configuration
        execution_config = self.get_execution_config()
        tasks = self.get_tasks_to_execute()
        print(f"Execute tasks: {', '.join(tasks)}")
        print(f"Verbose output: {self.is_verbose()}")
        print(f"Stop on error: {self.should_stop_on_error()}")
        
        # Base configuration
        base_config = self.get_base_config()
        print(f"Model: {base_config.get('model_name')}")
        print(f"Device: {base_config.get('device')}")
        
        # Module configurations
        if "circuit_tracing" in tasks:
            circuit_config = self.get_circuit_tracing_config()
            print(f"Circuit tracing: {circuit_config.get('num_samples')} samples, layers {circuit_config.get('layers_to_analyze_str')}")
        
        if "activation_analysis" in tasks:
            activation_config = self.get_activation_analysis_config()
            print(f"Activation analysis: top {activation_config.get('top_components')} components, {activation_config.get('max_prompts')} prompts")
        
        if "api_interpretation" in tasks:
            api_config = self.get_api_interpretation_config()
            print(f"API interpretation: {api_config.get('api_model')}, temperature {api_config.get('temperature')}")
        
        print("=" * 60)


# Global configuration manager instance
_config_manager = None

def get_config_manager(config_file: str = "config.yaml") -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

def reload_config(config_file: str = "config.yaml") -> ConfigManager:
    """Reload configuration"""
    global _config_manager
    _config_manager = ConfigManager(config_file)
    return _config_manager
