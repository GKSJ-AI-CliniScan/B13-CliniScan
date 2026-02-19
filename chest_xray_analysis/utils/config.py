"""
Configuration management utilities.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    with open(config_path, 'r') as f:
        if suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Raises:
        ValueError: If file format is not supported
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = config_path.suffix.lower()
    
    with open(config_path, 'w') as f:
        if suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")
