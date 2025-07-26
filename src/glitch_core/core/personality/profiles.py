"""
Personality profiles and drift patterns based on psychology research.
"""

from typing import Dict, List, Any


class PersonaConfig:
    """Defines base personality traits + evolution biases."""
    
    def __init__(
        self,
        traits: Dict[str, float],
        cognitive_biases: Dict[str, float],
        emotional_baselines: Dict[str, float],
        memory_patterns: Dict[str, float]
    ):
        self.traits = traits
        self.cognitive_biases = cognitive_biases
        self.emotional_baselines = emotional_baselines
        self.memory_patterns = memory_patterns
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "traits": self.traits,
            "cognitive_biases": self.cognitive_biases,
            "emotional_baselines": self.emotional_baselines,
            "memory_patterns": self.memory_patterns
        }


class DriftProfile:
    """Defines HOW personality evolves over time."""
    
    def __init__(
        self,
        evolution_rules: List[Dict[str, Any]],
        stability_metrics: Dict[str, Any],
        breakdown_conditions: List[Dict[str, Any]]
    ):
        self.evolution_rules = evolution_rules
        self.stability_metrics = stability_metrics
        self.breakdown_conditions = breakdown_conditions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "evolution_rules": self.evolution_rules,
            "stability_metrics": self.stability_metrics,
            "breakdown_conditions": self.breakdown_conditions
        }


# Predefined personality profiles
PERSONALITY_PROFILES = {
    "resilient_optimist": {
        "traits": {
            "openness": 0.7,
            "conscientiousness": 0.8,
            "extraversion": 0.6,
            "agreeableness": 0.7,
            "neuroticism": 0.3
        },
        "cognitive_biases": {
            "optimism_bias": 0.8,
            "confirmation_bias": 0.4,
            "anchoring_bias": 0.5
        },
        "emotional_baselines": {
            "joy": 0.7,
            "anxiety": 0.2,
            "anger": 0.2,
            "sadness": 0.2,
            "social_energy": 0.6
        },
        "memory_patterns": {
            "positive_recall": 0.8,
            "negative_recall": 0.3,
            "detail_retention": 0.6
        }
    },
    
    "anxious_overthinker": {
        "traits": {
            "openness": 0.6,
            "conscientiousness": 0.7,
            "extraversion": 0.3,
            "agreeableness": 0.5,
            "neuroticism": 0.8
        },
        "cognitive_biases": {
            "catastrophizing": 0.8,
            "confirmation_bias": 0.7,
            "overgeneralization": 0.6
        },
        "emotional_baselines": {
            "joy": 0.3,
            "anxiety": 0.7,
            "anger": 0.4,
            "sadness": 0.5,
            "social_energy": 0.3
        },
        "memory_patterns": {
            "positive_recall": 0.3,
            "negative_recall": 0.8,
            "detail_retention": 0.8
        }
    },
    
    "stoic_philosopher": {
        "traits": {
            "openness": 0.8,
            "conscientiousness": 0.6,
            "extraversion": 0.4,
            "agreeableness": 0.6,
            "neuroticism": 0.2
        },
        "cognitive_biases": {
            "stoic_dampening": 0.9,
            "rational_analysis": 0.8,
            "emotional_detachment": 0.7
        },
        "emotional_baselines": {
            "joy": 0.4,
            "anxiety": 0.1,
            "anger": 0.1,
            "sadness": 0.2,
            "social_energy": 0.4
        },
        "memory_patterns": {
            "positive_recall": 0.5,
            "negative_recall": 0.3,
            "detail_retention": 0.9
        }
    },
    
    "creative_volatile": {
        "traits": {
            "openness": 0.9,
            "conscientiousness": 0.4,
            "extraversion": 0.7,
            "agreeableness": 0.5,
            "neuroticism": 0.6
        },
        "cognitive_biases": {
            "creative_association": 0.9,
            "impulsivity": 0.7,
            "novelty_seeking": 0.8
        },
        "emotional_baselines": {
            "joy": 0.6,
            "anxiety": 0.4,
            "anger": 0.5,
            "sadness": 0.3,
            "social_energy": 0.7
        },
        "memory_patterns": {
            "positive_recall": 0.6,
            "negative_recall": 0.4,
            "detail_retention": 0.5
        }
    }
}


# Predefined drift profiles
DRIFT_PROFILES = {
    "resilient_optimist": {
        "evolution_rules": [
            {
                "type": "amplification",
                "target_emotion": "joy",
                "factor": 1.05,
                "condition": "positive_event"
            },
            {
                "type": "dampening",
                "target_emotion": "anxiety",
                "factor": 0.95,
                "condition": "any"
            },
            {
                "type": "oscillation",
                "target_emotion": "social_energy",
                "frequency": 15,
                "amplitude": 0.1
            }
        ],
        "stability_metrics": {
            "warning_threshold": 0.85,
            "recovery_rate": 0.8,
            "resilience_factor": 0.9
        },
        "breakdown_conditions": [
            {
                "type": "extreme_stress",
                "threshold": 0.95,
                "duration": 10
            }
        ]
    },
    
    "anxious_overthinker": {
        "evolution_rules": [
            {
                "type": "amplification",
                "target_emotion": "anxiety",
                "factor": 1.1,
                "condition": "stressful_event"
            },
            {
                "type": "rumination",
                "target_emotion": "sadness",
                "factor": 1.05,
                "condition": "negative_event"
            },
            {
                "type": "oscillation",
                "target_emotion": "anxiety",
                "frequency": 8,
                "amplitude": 0.15
            }
        ],
        "stability_metrics": {
            "warning_threshold": 0.7,
            "recovery_rate": 0.3,
            "resilience_factor": 0.4
        },
        "breakdown_conditions": [
            {
                "type": "anxiety_spiral",
                "threshold": 0.8,
                "duration": 5
            }
        ]
    },
    
    "stoic_philosopher": {
        "evolution_rules": [
            {
                "type": "dampening",
                "target_emotion": "all",
                "factor": 0.98,
                "condition": "any"
            },
            {
                "type": "rational_processing",
                "target_emotion": "anxiety",
                "factor": 0.9,
                "condition": "stressful_event"
            }
        ],
        "stability_metrics": {
            "warning_threshold": 0.95,
            "recovery_rate": 0.9,
            "resilience_factor": 0.95
        },
        "breakdown_conditions": [
            {
                "type": "emotional_bottleneck",
                "threshold": 0.98,
                "duration": 20
            }
        ]
    },
    
    "creative_volatile": {
        "evolution_rules": [
            {
                "type": "amplification",
                "target_emotion": "joy",
                "factor": 1.15,
                "condition": "creative_activity"
            },
            {
                "type": "amplification",
                "target_emotion": "anxiety",
                "factor": 1.1,
                "condition": "stressful_event"
            },
            {
                "type": "oscillation",
                "target_emotion": "all",
                "frequency": 12,
                "amplitude": 0.2
            }
        ],
        "stability_metrics": {
            "warning_threshold": 0.75,
            "recovery_rate": 0.6,
            "resilience_factor": 0.5
        },
        "breakdown_conditions": [
            {
                "type": "emotional_volatility",
                "threshold": 0.8,
                "duration": 8
            }
        ]
    }
}


def get_persona_config(profile_name: str) -> PersonaConfig:
    """Get a persona configuration by name."""
    if profile_name not in PERSONALITY_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")
    
    profile = PERSONALITY_PROFILES[profile_name]
    return PersonaConfig(
        traits=profile["traits"],
        cognitive_biases=profile["cognitive_biases"],
        emotional_baselines=profile["emotional_baselines"],
        memory_patterns=profile["memory_patterns"]
    )


def get_drift_profile(profile_name: str) -> DriftProfile:
    """Get a drift profile by name."""
    if profile_name not in DRIFT_PROFILES:
        raise ValueError(f"Unknown drift profile: {profile_name}")
    
    profile = DRIFT_PROFILES[profile_name]
    return DriftProfile(
        evolution_rules=profile["evolution_rules"],
        stability_metrics=profile["stability_metrics"],
        breakdown_conditions=profile["breakdown_conditions"]
    )


def list_available_profiles() -> List[str]:
    """List all available personality profiles."""
    return list(PERSONALITY_PROFILES.keys())


def list_available_drift_profiles() -> List[str]:
    """List all available drift profiles."""
    return list(DRIFT_PROFILES.keys()) 