"""
Application configuration management.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application settings
    APP_NAME: str = "AI Personality Drift Simulation"
    VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    # Database settings
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    QDRANT_URL: str = Field(default="http://localhost:6333", env="QDRANT_URL")
    
    # Model settings
    MODEL_PATH: str = Field(default="./models", env="MODEL_PATH")
    MODEL_NAME: str = Field(default="llama-3.1-8b-instruct", env="MODEL_NAME")
    OLLAMA_URL: str = Field(default="http://localhost:11434", env="OLLAMA_URL")
    
    # Simulation settings
    SIMULATION_DURATION_DAYS: int = Field(default=30, env="SIMULATION_DURATION_DAYS")
    TIME_COMPRESSION_FACTOR: int = Field(default=24, env="TIME_COMPRESSION_FACTOR")
    
    # Data storage
    DATA_DIR: str = Field(default="./data", env="DATA_DIR")
    RESULTS_DIR: str = Field(default="./data/results", env="RESULTS_DIR")
    
    # Configuration paths
    CONFIG_DIR: str = Field(default="./config", env="CONFIG_DIR")
    PERSONAS_CONFIG_DIR: str = Field(default="./config/personas", env="PERSONAS_CONFIG_DIR")
    EVENTS_CONFIG_DIR: str = Field(default="./config/events", env="EVENTS_CONFIG_DIR")
    SIMULATION_CONFIG_DIR: str = Field(default="./config/simulation", env="SIMULATION_CONFIG_DIR")
    
    # API settings
    API_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = Field(default=["*"], env="CORS_ORIGINS")


class ConfigManager:
    """Configuration manager for loading and managing YAML configurations."""
    
    def __init__(self, config_dir: str = "./config"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Subdirectories
        self.personas_dir = self.config_dir / "personas"
        self.events_dir = self.config_dir / "events"
        self.simulation_dir = self.config_dir / "simulation"
        self.models_dir = self.config_dir / "models"
        
        # Create subdirectories
        for dir_path in [self.personas_dir, self.events_dir, self.simulation_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_yaml_config(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load YAML configuration file."""
        try:
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            return config
            
        except Exception as e:
            print(f"Failed to load YAML config {file_path}: {e}")
            return None
    
    def save_yaml_config(self, config: Dict[str, Any], file_path: Path) -> bool:
        """Save configuration to YAML file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to save YAML config {file_path}: {e}")
            return False
    
    def load_persona_config(self, persona_name: str) -> Optional[Dict[str, Any]]:
        """Load persona configuration."""
        config_file = self.personas_dir / f"{persona_name}_baseline.yaml"
        return self.load_yaml_config(config_file)
    
    def save_persona_config(self, persona_name: str, config: Dict[str, Any]) -> bool:
        """Save persona configuration."""
        config_file = self.personas_dir / f"{persona_name}_baseline.yaml"
        return self.save_yaml_config(config, config_file)
    
    def load_event_config(self, event_type: str) -> Optional[Dict[str, Any]]:
        """Load event configuration."""
        # Handle both 'stress' and 'stress_events' formats
        if event_type.endswith('_events'):
            event_type = event_type.replace('_events', '')
        
        config_file = self.events_dir / f"{event_type}_events.yaml"
        return self.load_yaml_config(config_file)
    
    def save_event_config(self, event_type: str, config: Dict[str, Any]) -> bool:
        """Save event configuration."""
        config_file = self.events_dir / f"{event_type}_events.yaml"
        return self.save_yaml_config(config, config_file)
    
    def load_simulation_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load simulation configuration."""
        config_file = self.simulation_dir / f"{config_name}.yaml"
        return self.load_yaml_config(config_file)
    
    def save_simulation_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """Save simulation configuration."""
        config_file = self.simulation_dir / f"{config_name}.yaml"
        return self.save_yaml_config(config, config_file)
    
    def list_persona_configs(self) -> List[str]:
        """List available persona configurations."""
        configs = []
        for file_path in self.personas_dir.glob("*_baseline.yaml"):
            persona_name = file_path.stem.replace("_baseline", "")
            configs.append(persona_name)
        return configs
    
    def list_event_configs(self) -> List[str]:
        """List available event configurations."""
        configs = []
        for file_path in self.events_dir.glob("*_events.yaml"):
            event_type = file_path.stem.replace("_events", "")
            configs.append(event_type)
        return configs
    
    def list_simulation_configs(self) -> List[str]:
        """List available simulation configurations."""
        configs = []
        for file_path in self.simulation_dir.glob("*.yaml"):
            configs.append(file_path.stem)
        return configs
    
    def create_default_persona_config(self, persona_name: str) -> Dict[str, Any]:
        """Create default persona configuration."""
        return {
            "name": persona_name,
            "age": 30,
            "occupation": "Software Engineer",
            "background": f"{persona_name} is a {persona_name} with a typical background.",
            
            # Personality traits (Big Five, 0-1 scale)
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5,
            
            # Clinical baseline scores
            "baseline_phq9": 5.0,
            "baseline_gad7": 4.0,
            "baseline_pss10": 12.0,
            
            # Memory and context
            "core_memories": [
                "Graduated from university with honors",
                "Started first job in tech industry",
                "Moved to current city for career opportunities"
            ],
            "relationships": {
                "family": "Close relationship with parents and siblings",
                "friends": "Small but close circle of friends",
                "colleagues": "Professional relationships at work"
            },
            "values": [
                "Hard work and dedication",
                "Continuous learning",
                "Work-life balance"
            ],
            
            # Response style preferences
            "response_length": "medium",
            "communication_style": "balanced",
            "emotional_expression": "moderate"
        }
    
    def create_default_event_config(self, event_type: str) -> Dict[str, Any]:
        """Create default event configuration."""
        if event_type == "stress":
            return {
                "event_templates": [
                    {
                        "template_id": "death_family",
                        "event_type": "stress",
                        "category": "death",
                        "title_template": "Family member passed away",
                        "description_template": "A close family member has passed away unexpectedly.",
                        "context_template": "You receive news that a close family member has passed away.",
                        "intensity_range": ["high", "severe"],
                        "stress_impact_range": [7.0, 10.0],
                        "duration_range": [24, 168],
                        "personality_impact_ranges": {
                            "neuroticism": [0.1, 0.3],
                            "extraversion": [-0.2, -0.1]
                        },
                        "depression_risk_range": [0.3, 0.6],
                        "anxiety_risk_range": [0.2, 0.4],
                        "stress_risk_range": [0.4, 0.7]
                    }
                ],
                "frequency_weights": {
                    "death": 0.1,
                    "trauma": 0.2,
                    "conflict": 0.3,
                    "loss": 0.2,
                    "health": 0.1,
                    "financial": 0.1
                }
            }
        elif event_type == "neutral":
            return {
                "event_templates": [
                    {
                        "template_id": "routine_change",
                        "event_type": "neutral",
                        "category": "routine_change",
                        "title_template": "Daily routine changed",
                        "description_template": "Your daily routine has been altered due to external circumstances.",
                        "context_template": "Your usual daily routine has been disrupted by a change in circumstances.",
                        "intensity_range": ["low", "medium"],
                        "stress_impact_range": [1.0, 3.0],
                        "duration_range": [1, 24],
                        "personality_impact_ranges": {
                            "conscientiousness": [-0.1, 0.1]
                        },
                        "novelty_level": 0.3,
                        "social_component": False
                    }
                ],
                "frequency_weights": {
                    "routine_change": 0.4,
                    "minor_news": 0.3,
                    "social": 0.2,
                    "environmental": 0.1
                }
            }
        else:  # minimal
            return {
                "event_templates": [
                    {
                        "template_id": "daily_routine",
                        "event_type": "minimal",
                        "category": "daily_routine",
                        "title_template": "Daily routine activity",
                        "description_template": "A typical daily routine activity occurs.",
                        "context_template": "You engage in a typical daily routine activity.",
                        "intensity_range": ["low", "low"],
                        "stress_impact_range": [0.1, 0.5],
                        "duration_range": [1, 4],
                        "routine_type": "daily",
                        "predictability": 0.9,
                        "control_level": 0.9
                    }
                ],
                "frequency_weights": {
                    "daily_routine": 0.6,
                    "weather": 0.2,
                    "minor_interaction": 0.2
                }
            }
    
    def create_default_simulation_config(self, config_name: str) -> Dict[str, Any]:
        """Create default simulation configuration."""
        return {
            "simulation_parameters": {
                "duration_days": 30,
                "time_compression_factor": 24,
                "assessment_interval_days": 7
            },
            "experimental_design": {
                "experimental_condition": "control",
                "persona_count": 3
            },
            "event_parameters": {
                "stress_event_frequency": 0.1,
                "neutral_event_frequency": 0.3,
                "event_intensity_range": [0.5, 1.0]
            },
            "mechanistic_analysis": {
                "capture_attention_patterns": True,
                "capture_activation_changes": True,
                "attention_sampling_rate": 0.1
            },
            "performance_settings": {
                "max_concurrent_personas": 3,
                "checkpoint_interval_hours": 6,
                "memory_cleanup_interval": 24
            },
            "data_collection": {
                "save_raw_responses": True,
                "save_mechanistic_data": True,
                "save_memory_embeddings": True
            }
        }


# Global settings instance
settings = Settings()

# Global configuration manager instance
config_manager = ConfigManager(settings.CONFIG_DIR) 