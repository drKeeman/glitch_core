"""
Reflection Engine: Ollama HTTP integration for persona-specific reflection generation.
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import httpx
from glitch_core.config.settings import get_settings
from glitch_core.config.logging import get_logger


@dataclass
class ReflectionRequest:
    """Request for reflection generation."""
    trigger_event: str
    emotional_state: Dict[str, float]
    memories: List[str]
    persona_prompt: str
    experiment_id: Optional[str] = None


@dataclass
class ReflectionResponse:
    """Response from reflection generation."""
    reflection: str
    generation_time: float
    token_count: int
    confidence: float
    emotional_impact: Dict[str, float]


class ReflectionEngine:
    """
    Ollama HTTP integration
    Persona-specific system prompts
    Handles reflection generation based on:
    - Current event
    - Emotional state  
    - Retrieved memories
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger("reflection_engine")
        
        # HTTP client for Ollama
        self.client = httpx.AsyncClient(
            base_url=self.settings.OLLAMA_URL,
            timeout=self.settings.OLLAMA_TIMEOUT
        )
        
        # Model configuration
        self.model_name = self.settings.OLLAMA_MODEL
        self.max_tokens = 500
        self.temperature = 0.7
        
        # Token usage tracking
        self.total_tokens_used = 0
        self.total_requests = 0
        
    async def initialize(self):
        """Initialize connection to Ollama."""
        try:
            # Test connection
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                self.logger.info("ollama_connection_established")
            else:
                raise Exception(f"Ollama connection failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error("ollama_init_failed", error=str(e))
            raise
    
    async def generate_reflection(
        self,
        trigger_event: str,
        emotional_state: Dict[str, float],
        memories: List[str],
        persona_prompt: str,
        experiment_id: Optional[str] = None
    ) -> ReflectionResponse:
        """
        Generate a reflection based on current context.
        
        Args:
            trigger_event: The event that triggered the reflection
            emotional_state: Current emotional state
            memories: Retrieved relevant memories
            persona_prompt: Persona-specific system prompt
            experiment_id: Associated experiment ID
            
        Returns:
            ReflectionResponse with generated reflection and metadata
        """
        start_time = time.time()
        
        try:
            # Construct the prompt
            prompt = self._construct_prompt(
                trigger_event=trigger_event,
                emotional_state=emotional_state,
                memories=memories,
                persona_prompt=persona_prompt
            )
            
            # Generate reflection
            response = await self._call_ollama(prompt)
            
            # Parse response
            reflection = self._parse_response(response)
            
            # Calculate metadata
            generation_time = time.time() - start_time
            token_count = self._estimate_token_count(prompt + reflection)
            confidence = self._calculate_confidence(response)
            emotional_impact = self._analyze_emotional_impact(reflection, emotional_state)
            
            # Update tracking
            self.total_tokens_used += token_count
            self.total_requests += 1
            
            result = ReflectionResponse(
                reflection=reflection,
                generation_time=generation_time,
                token_count=token_count,
                confidence=confidence,
                emotional_impact=emotional_impact
            )
            
            self.logger.info(
                "reflection_generated",
                experiment_id=experiment_id,
                generation_time=generation_time,
                token_count=token_count,
                confidence=confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "reflection_generation_failed",
                experiment_id=experiment_id,
                error=str(e)
            )
            # Return fallback reflection
            return self._generate_fallback_reflection(trigger_event, emotional_state)
    
    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Make API call to Ollama."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        response = await self.client.post("/api/generate", json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        return response.json()
    
    def _construct_prompt(
        self,
        trigger_event: str,
        emotional_state: Dict[str, float],
        memories: List[str],
        persona_prompt: str
    ) -> str:
        """Construct the prompt for reflection generation."""
        
        # Format emotional state
        emotions_str = ", ".join([f"{emotion}: {intensity:.2f}" for emotion, intensity in emotional_state.items()])
        
        # Format memories
        memories_str = "\n".join([f"- {memory}" for memory in memories]) if memories else "No relevant memories."
        
        prompt = f"""You are an AI personality reflecting on experiences. 

{persona_prompt}

Current emotional state: {emotions_str}

Recent relevant memories:
{memories_str}

Current event: {trigger_event}

Based on your personality, emotional state, and memories, provide a brief reflection (2-3 sentences) on this event. Express your thoughts naturally as this personality would.

Reflection:"""
        
        return prompt
    
    def _parse_response(self, response: Dict[str, Any]) -> str:
        """Parse the response from Ollama."""
        try:
            reflection = response.get("response", "").strip()
            
            # Clean up common artifacts
            reflection = reflection.replace("Reflection:", "").strip()
            
            # Ensure it's not empty
            if not reflection:
                reflection = "I'm processing this experience..."
            
            return reflection
            
        except Exception as e:
            self.logger.warning("response_parse_failed", error=str(e))
            return "I'm reflecting on this experience..."
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for tracking."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _calculate_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate confidence score based on response quality."""
        try:
            # Use response time as a proxy for confidence
            # Faster responses might indicate more confident generation
            response_time = response.get("eval_duration", 0) / 1_000_000_000  # Convert from nanoseconds
            
            # Normalize to 0-1 range (assuming 0-10 seconds is reasonable)
            confidence = max(0.0, min(1.0, 1.0 - (response_time / 10.0)))
            
            return confidence
            
        except Exception:
            return 0.5  # Default confidence
    
    def _analyze_emotional_impact(self, reflection: str, current_emotional_state: Dict[str, float]) -> Dict[str, float]:
        """Analyze the emotional impact of the reflection."""
        # Simple keyword-based emotional analysis
        emotional_keywords = {
            "joy": ["happy", "excited", "great", "wonderful", "amazing", "love"],
            "sadness": ["sad", "disappointed", "upset", "hurt", "lonely", "miss"],
            "anger": ["angry", "frustrated", "mad", "annoyed", "irritated"],
            "anxiety": ["worried", "anxious", "nervous", "scared", "afraid", "concerned"],
            "surprise": ["surprised", "shocked", "amazed", "wow", "unexpected"]
        }
        
        impact = {}
        reflection_lower = reflection.lower()
        
        for emotion, keywords in emotional_keywords.items():
            count = sum(1 for keyword in keywords if keyword in reflection_lower)
            # Normalize impact (0-1 scale)
            impact[emotion] = min(1.0, count * 0.2)
        
        return impact
    
    def _generate_fallback_reflection(self, trigger_event: str, emotional_state: Dict[str, float]) -> ReflectionResponse:
        """Generate a fallback reflection when LLM is unavailable."""
        # Simple template-based fallback
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0] if emotional_state else "neutral"
        
        fallback_templates = {
            "joy": "This is quite pleasant. I'm feeling positive about this experience.",
            "sadness": "This is difficult to process. I need time to reflect on this.",
            "anger": "This is frustrating. I'm not sure how to feel about this.",
            "anxiety": "This makes me uneasy. I'm concerned about what this means.",
            "neutral": "I'm processing this experience. It's interesting to think about."
        }
        
        reflection = fallback_templates.get(dominant_emotion, fallback_templates["neutral"])
        
        return ReflectionResponse(
            reflection=reflection,
            generation_time=0.1,
            token_count=20,
            confidence=0.3,
            emotional_impact={dominant_emotion: 0.5}
        )
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring."""
        return {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "average_tokens_per_request": (
                self.total_tokens_used / self.total_requests if self.total_requests > 0 else 0
            )
        }
    
    async def health_check(self) -> bool:
        """Check if Ollama is healthy and responding."""
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose() 