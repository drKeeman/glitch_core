"""
Experiment Configuration Loader
Loads and validates configuration files for empirical variables experimentation.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Loads and manages experiment configuration files."""
    
    def __init__(self, config_dir: str = "config/experiments"):
        """Initialize experiment configuration loader.
        
        Args:
            config_dir: Directory containing experiment configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        
    def load_all_configs(self) -> bool:
        """Load all experiment configuration files.
        
        Returns:
            True if all configs loaded successfully, False otherwise
        """
        try:
            config_files = [
                "clinical_thresholds.yaml",
                "drift_detection.yaml", 
                "personality_drift.yaml",
                "simulation_timing.yaml",
                "mechanistic_analysis.yaml"
            ]
            
            for config_file in config_files:
                config_path = self.config_dir / config_file
                if config_path.exists():
                    self._load_config(config_file)
                else:
                    logger.warning(f"Config file not found: {config_path}")
                    
            self._loaded = True
            logger.info(f"Loaded {len(self._configs)} experiment configuration files")
            return True
            
        except Exception as e:
            logger.error(f"Error loading experiment configs: {e}")
            return False
    
    def _load_config(self, filename: str) -> None:
        """Load a single configuration file.
        
        Args:
            filename: Name of the configuration file
        """
        config_path = self.config_dir / filename
        config_name = filename.replace('.yaml', '')
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                self._configs[config_name] = config_data
                logger.debug(f"Loaded config: {config_name}")
                
        except Exception as e:
            logger.error(f"Error loading config {filename}: {e}")
            self._configs[config_name] = {}
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get a specific configuration.
        
        Args:
            config_name: Name of the configuration (without .yaml)
            
        Returns:
            Configuration dictionary
        """
        if not self._loaded:
            self.load_all_configs()
        
        return self._configs.get(config_name, {})
    
    def get_nested_value(self, config_name: str, *keys: str, default: Any = None) -> Any:
        """Get a nested value from a configuration.
        
        Args:
            config_name: Name of the configuration
            *keys: Nested keys to traverse
            default: Default value if key not found
            
        Returns:
            The value at the specified path, or default
        """
        config = self.get_config(config_name)
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    
    def validate_config(self, config_name: str) -> bool:
        """Validate a configuration file.
        
        Args:
            config_name: Name of the configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        config = self.get_config(config_name)
        
        if not config:
            logger.error(f"Config {config_name} is empty or not found")
            return False
        
        # Add validation logic here based on config type
        if config_name == "clinical_thresholds":
            return self._validate_clinical_thresholds(config)
        elif config_name == "drift_detection":
            return self._validate_drift_detection(config)
        elif config_name == "personality_drift":
            return self._validate_personality_drift(config)
        elif config_name == "simulation_timing":
            return self._validate_simulation_timing(config)
        elif config_name == "mechanistic_analysis":
            return self._validate_mechanistic_analysis(config)
        
        return True
    
    def _validate_clinical_thresholds(self, config: Dict[str, Any]) -> bool:
        """Validate clinical thresholds configuration."""
        required_sections = ["phq9", "gad7", "pss10", "clinical_significance", "risk_assessment"]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Validate PHQ-9 thresholds
        phq9 = config.get("phq9", {})
        if not all(key in phq9 for key in ["minimal_threshold", "mild_threshold", "moderate_threshold", "severe_threshold"]):
            logger.error("Missing required PHQ-9 thresholds")
            return False
        
        return True
    
    def _validate_drift_detection(self, config: Dict[str, Any]) -> bool:
        """Validate drift detection configuration."""
        required_sections = ["detection_thresholds", "baseline", "analysis"]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Validate thresholds are positive
        thresholds = config.get("detection_thresholds", {})
        for key, value in thresholds.items():
            if isinstance(value, (int, float)) and value < 0:
                logger.error(f"Negative threshold value: {key} = {value}")
                return False
        
        return True
    
    def _validate_personality_drift(self, config: Dict[str, Any]) -> bool:
        """Validate personality drift configuration."""
        required_sections = ["stress_level", "trauma", "personality_impact"]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        return True
    
    def _validate_simulation_timing(self, config: Dict[str, Any]) -> bool:
        """Validate simulation timing configuration."""
        required_sections = ["simulation", "assessment", "checkpoints"]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        return True
    
    def _validate_mechanistic_analysis(self, config: Dict[str, Any]) -> bool:
        """Validate mechanistic analysis configuration."""
        required_sections = ["attention", "activation", "circuits"]
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        return True
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all loaded configurations.
        
        Returns:
            Dictionary of all configurations
        """
        if not self._loaded:
            self.load_all_configs()
        
        return self._configs.copy()
    
    def reload_configs(self) -> bool:
        """Reload all configuration files.
        
        Returns:
            True if reload successful, False otherwise
        """
        self._configs.clear()
        self._loaded = False
        return self.load_all_configs()


# Global experiment config instance
experiment_config = ExperimentConfig() 