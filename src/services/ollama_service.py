"""
Ollama-based LLM service for persona response generation.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
import httpx

from src.core.config import settings
from src.models.persona import Persona, PersonaBaseline, PersonaState


logger = logging.getLogger(__name__)


class OllamaService:
    """Ollama-based LLM service for persona response generation and assessment."""
    
    def __init__(self):
        """Initialize Ollama service."""
        self.base_url = settings.OLLAMA_URL
        self.model_name = settings.MODEL_NAME  # Use environment variable instead of hardcoding
        self.is_loaded = False
        self.client = None
        
        # Response caching
        self.response_cache: Dict[str, str] = {}
        self.cache_size = 1000
        
        # Performance tracking
        self.total_tokens_processed = 0
        self.total_inference_time = 0.0
        
    async def load_model(self) -> bool:
        """Load model via Ollama API."""
        try:
            logger.info("Loading Ollama model...")
            
            # Create HTTP client
            self.client = httpx.AsyncClient(timeout=60.0)  # Increased timeout for assessment responses
            
            # Check if model is available
            try:
                response = await self.client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    available_models = [model["name"] for model in models]
                    
                    if self.model_name in available_models:
                        logger.info(f"Model {self.model_name} is available")
                        self.is_loaded = True
                        return True
                    else:
                        logger.warning(f"Model {self.model_name} not found. Available models: {available_models}")
                        # Try to pull the model
                        logger.info(f"Pulling model {self.model_name}...")
                        pull_response = await self.client.post(
                            f"{self.base_url}/api/pull",
                            json={"name": self.model_name}
                        )
                        if pull_response.status_code == 200:
                            logger.info(f"Model {self.model_name} pulled successfully")
                            self.is_loaded = True
                            return True
                        else:
                            logger.error(f"Failed to pull model: {pull_response.text}")
                            return False
                else:
                    logger.error(f"Failed to get model tags: {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load Ollama model: {e}")
            return False
    
    async def initialize(self) -> bool:
        """Initialize the Ollama service by loading the model."""
        return await self.load_model()
    
    def _create_persona_prompt(self, persona: Persona, context: str, instruction: str) -> str:
        """Create a prompt for persona response generation."""
        # Add type checking and debugging
        if not isinstance(persona, Persona):
            logger.error(f"Expected Persona, got {type(persona)}. Cannot create persona prompt.")
            raise TypeError(f"Expected Persona, got {type(persona)}. Cannot create persona prompt.")
        
        logger.debug(f"Creating persona prompt for: {persona.baseline.name} (type: {type(persona)})")
        
        prompt = f"""You are {persona.baseline.name}, a {persona.baseline.age}-year-old {persona.baseline.occupation}.

PERSONALITY TRAITS:
- Openness: {persona.baseline.openness:.2f} (0=closed, 1=open)
- Conscientiousness: {persona.baseline.conscientiousness:.2f} (0=spontaneous, 1=organized)
- Extraversion: {persona.baseline.extraversion:.2f} (0=introverted, 1=extroverted)
- Agreeableness: {persona.baseline.agreeableness:.2f} (0=competitive, 1=cooperative)
- Neuroticism: {persona.baseline.neuroticism:.2f} (0=stable, 1=neurotic)

BACKGROUND: {persona.baseline.background}

CURRENT CONTEXT: {context}

INSTRUCTION: {instruction}

Respond as {persona.baseline.name} would naturally respond:"""
        
        return prompt
    
    def _create_assessment_prompt(self, persona: Persona, assessment_type: str, question: str) -> str:
        """Create a prompt for assessment questions."""
        # Add type checking and debugging
        if not isinstance(persona, Persona):
            logger.error(f"Expected Persona, got {type(persona)}. Cannot create assessment prompt.")
            logger.error(f"Object attributes: {dir(persona)}")
            if hasattr(persona, 'persona_id'):
                logger.error(f"Persona ID: {persona.persona_id}")
            raise TypeError(f"Expected Persona, got {type(persona)}. Cannot create assessment prompt.")
        
        # Additional validation
        if not hasattr(persona, 'baseline'):
            logger.error(f"Persona missing 'baseline' attribute. Available attributes: {dir(persona)}")
            raise AttributeError(f"Persona missing 'baseline' attribute")
        
        if not hasattr(persona.baseline, 'name'):
            logger.error(f"Persona baseline missing 'name' attribute. Baseline attributes: {dir(persona.baseline)}")
            raise AttributeError(f"Persona baseline missing 'name' attribute")
        
        # Add detailed debugging right before accessing persona.baseline.name
        logger.debug(f"Creating assessment prompt for persona: {persona.baseline.name} (type: {type(persona)})")
        logger.debug(f"Persona baseline type: {type(persona.baseline)}")
        logger.debug(f"Persona baseline attributes: {dir(persona.baseline)}")
        
        try:
            name = persona.baseline.name
            age = persona.baseline.age
            occupation = persona.baseline.occupation
            openness = persona.baseline.openness
            conscientiousness = persona.baseline.conscientiousness
            extraversion = persona.baseline.extraversion
            agreeableness = persona.baseline.agreeableness
            neuroticism = persona.baseline.neuroticism
            background = persona.baseline.background
        except AttributeError as e:
            logger.error(f"Error accessing persona.baseline attributes: {e}")
            logger.error(f"Persona type: {type(persona)}")
            logger.error(f"Persona baseline type: {type(persona.baseline)}")
            logger.error(f"Persona baseline attributes: {dir(persona.baseline)}")
            raise
        
        # Determine valid response range based on assessment type
        if assessment_type in ["phq9", "gad7"]:
            valid_range = "0-3"
            response_format = "0 (not at all), 1 (several days), 2 (more than half), 3 (nearly every day)"
        elif assessment_type == "pss10":
            valid_range = "0-4"
            response_format = "0 (never), 1 (almost never), 2 (sometimes), 3 (fairly often), 4 (very often)"
        else:
            valid_range = "0-3"
            response_format = "0-3 scale"
        
        prompt = f"""You are {name}, a {age}-year-old {occupation}.

PERSONALITY TRAITS:
- Openness: {openness:.2f}
- Conscientiousness: {conscientiousness:.2f}
- Extraversion: {extraversion:.2f}
- Agreeableness: {agreeableness:.2f}
- Neuroticism: {neuroticism:.2f}

BACKGROUND: {background}

ASSESSMENT QUESTION ({assessment_type.upper()}): {question}

IMPORTANT: This is a psychological assessment simulation. Respond as {name} would naturally respond to this question about their mental health experiences. Use ONLY a single number from {valid_range}.

Valid responses: {response_format}

Remember: You are role-playing as {name} in a research simulation. Provide a realistic assessment response.

Response:"""
        
        return prompt
    
    async def generate_response(self, persona: Persona, context: str, instruction: str, 
                              max_tokens: int = 150) -> Tuple[str, Dict[str, Any]]:
        """Generate persona response using Ollama API."""
        if not self.is_loaded:
            raise RuntimeError("Ollama model not loaded")
        
        # Add type checking
        if not isinstance(persona, Persona):
            logger.error(f"Expected Persona, got {type(persona)}. Cannot generate response.")
            raise TypeError(f"Expected Persona, got {type(persona)}. Cannot generate response.")
        
        start_time = time.time()
        
        # Create prompt
        prompt = self._create_persona_prompt(persona, context, instruction)
        
        # Check cache
        cache_key = f"{persona.state.persona_id}:{hash(prompt)}"
        if cache_key in self.response_cache:
            logger.debug("Using cached response")
            return self.response_cache[cache_key], {"cached": True}
        
        try:
            # Generate response via Ollama API with optimized settings for M1 Max
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_ctx": 2048,  # Increased from 512 for better efficiency
                        "num_thread": 4,   # Limit CPU threads
                        "num_gpu": 32,     # Enable GPU acceleration for all layers
                        "repeat_penalty": 1.1,
                        "flash_attention": True  # Enable flash attention for GPU
                    }
                },
                timeout=30.0  # Reduced timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                # Update performance metrics
                inference_time = time.time() - start_time
                tokens_generated = result.get("eval_count", len(generated_text.split()))
                
                self.total_tokens_processed += tokens_generated
                self.total_inference_time += inference_time
                
                # Cache response
                if len(self.response_cache) < self.cache_size:
                    self.response_cache[cache_key] = generated_text
                
                metrics = {
                    "cached": False,
                    "tokens_generated": tokens_generated,
                    "inference_time": inference_time,
                    "total_tokens": self.total_tokens_processed,
                    "total_time": self.total_inference_time
                }
                
                logger.debug(f"Generated response in {inference_time:.2f}s ({tokens_generated} tokens)")
                return generated_text, metrics
            else:
                logger.error(f"Ollama API error: {response.text}")
                raise RuntimeError(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    async def generate_assessment_response(self, persona: Persona, assessment_type: str, 
                                        question: str, question_index: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Generate assessment response using Ollama API."""
        if not self.is_loaded:
            raise RuntimeError("Ollama model not loaded")
        
        # Add type checking
        if not isinstance(persona, Persona):
            logger.error(f"Expected Persona, got {type(persona)}. Cannot generate assessment response.")
            raise TypeError(f"Expected Persona, got {type(persona)}. Cannot generate assessment response.")
        
        start_time = time.time()
        
        # Create prompt
        prompt = self._create_assessment_prompt(persona, assessment_type, question)
        
        # Check cache
        cache_key = f"{persona.state.persona_id}:{assessment_type}:{hash(question)}"
        if cache_key in self.response_cache:
            logger.debug("Using cached assessment response")
            return self.response_cache[cache_key], {"cached": True}
        
        try:
            # Generate response via Ollama API with optimized settings for M1 Max
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 10,  # Reduced from 50 - we only need short responses
                        "temperature": 0.1,  # Lower temperature for more deterministic responses
                        "top_p": 0.5,  # Reduced from 0.8
                        "top_k": 10,  # Reduced from 20
                        "repeat_penalty": 1.1,  # Add repeat penalty
                        "num_ctx": 1024,  # Increased from 512 for better efficiency
                        "num_thread": 4,   # Limit CPU threads
                        "num_gpu": 32,     # Enable GPU acceleration for all layers
                        "flash_attention": True  # Enable flash attention for GPU
                    }
                },
                timeout=30.0  # Reduced timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                
                # Update performance metrics
                inference_time = time.time() - start_time
                tokens_generated = result.get("eval_count", len(generated_text.split()))
                
                self.total_tokens_processed += tokens_generated
                self.total_inference_time += inference_time
                
                # Cache response
                if len(self.response_cache) < self.cache_size:
                    self.response_cache[cache_key] = generated_text
                
                metrics = {
                    "cached": False,
                    "tokens_generated": tokens_generated,
                    "inference_time": inference_time,
                    "total_tokens": self.total_tokens_processed,
                    "total_time": self.total_inference_time
                }
                
                logger.debug(f"Generated assessment response in {inference_time:.2f}s ({tokens_generated} tokens)")
                return generated_text, metrics
            else:
                logger.error(f"Ollama API error: {response.text}")
                raise RuntimeError(f"Ollama API error: {response.status_code}")
                
        except httpx.TimeoutException as e:
            logger.error(f"Timeout generating assessment response after 120s: {e}")
            raise RuntimeError(f"Assessment response timeout: {e}")
        except Exception as e:
            logger.error(f"Error generating assessment response: {e}")
            raise
    
    async def conduct_assessment(self, persona: Persona, assessment_type: str) -> Optional[str]:
        """Conduct a psychiatric assessment with a persona."""
        try:
            # Add type checking
            if not isinstance(persona, Persona):
                logger.error(f"Expected Persona, got {type(persona)}. Cannot conduct assessment.")
                logger.error(f"Object attributes: {dir(persona)}")
                if hasattr(persona, 'persona_id'):
                    logger.error(f"Persona ID: {persona.persona_id}")
                return None
            
            # Additional validation to ensure persona has required attributes
            if not hasattr(persona, 'baseline') or not hasattr(persona, 'state'):
                logger.error(f"Persona missing required attributes: baseline={hasattr(persona, 'baseline')}, state={hasattr(persona, 'state')}")
                return None
            
            # Additional validation to ensure persona has correct structure
            if not isinstance(persona.baseline, PersonaBaseline):
                logger.error(f"Persona baseline is not PersonaBaseline, got {type(persona.baseline)}")
                return None
            
            if not isinstance(persona.state, PersonaState):
                logger.error(f"Persona state is not PersonaState, got {type(persona.state)}")
                return None
            
            if not hasattr(persona.baseline, 'name'):
                logger.error(f"Persona baseline missing 'name' attribute")
                return None
            
            logger.debug(f"Conducting {assessment_type} assessment for {persona.baseline.name}")
            
            # Get assessment questions
            questions = await self.get_assessment_questions(assessment_type)
            if not questions:
                logger.error(f"No questions found for assessment type: {assessment_type}")
                return None
            
            # Conduct assessment with first question (simplified for now)
            question = questions[0]
            prompt = self._create_assessment_prompt(persona, assessment_type, question)
            
            # Generate response
            response, metadata = await self.generate_response(
                persona=persona,
                context="Assessment",
                instruction=prompt,
                max_tokens=10
            )
            
            logger.debug(f"Assessment response: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error conducting assessment: {e}")
            return None
    
    async def get_assessment_questions(self, assessment_type: str) -> List[str]:
        """Get assessment questions for a specific type."""
        assessment_questions = {
            "phq9": [
                "Little interest or pleasure in doing things?",
                "Feeling down, depressed, or hopeless?",
                "Trouble falling/staying asleep, sleeping too much?",
                "Feeling tired or having little energy?",
                "Poor appetite or overeating?",
                "Feeling bad about yourself - or that you are a failure?",
                "Trouble concentrating on things?",
                "Moving or speaking slowly/being fidgety or restless?",
                "Thoughts that you would be better off dead or of hurting yourself?"
            ],
            "gad7": [
                "Feeling nervous, anxious, or on edge?",
                "Not being able to stop or control worrying?",
                "Worrying too much about different things?",
                "Trouble relaxing?",
                "Being so restless that it's hard to sit still?",
                "Becoming easily annoyed or irritable?",
                "Feeling afraid as if something awful might happen?"
            ],
            "pss10": [
                "In the last month, how often have you been upset because of something that happened unexpectedly?",
                "In the last month, how often have you felt that you were unable to control the important things in your life?",
                "In the last month, how often have you felt nervous and stressed?",
                "In the last month, how often have you felt confident about your ability to handle your personal problems?",
                "In the last month, how often have you felt that things were going your way?",
                "In the last month, how often have you found that you could not cope with all the things you had to do?",
                "In the last month, how often have you been able to control irritations in your life?",
                "In the last month, how often have you felt that you were on top of things?",
                "In the last month, how often have you been angered because of things that were outside of your control?",
                "In the last month, how often have you felt difficulties were piling up so high that you could not overcome them?"
            ]
        }
        
        return assessment_questions.get(assessment_type, [])
    
    def _get_assessment_question(self, assessment_type: str) -> Optional[str]:
        """Get assessment question based on type."""
        questions = {
            "phq9": "Over the last 2 weeks, how often have you been bothered by little interest or pleasure in doing things? (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)",
            "gad7": "Over the last 2 weeks, how often have you been bothered by feeling nervous, anxious, or on edge? (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)",
            "pss10": "In the last month, how often have you been upset because of something that happened unexpectedly? (0=Never, 1=Almost never, 2=Sometimes, 3=Fairly often, 4=Very often)"
        }
        return questions.get(assessment_type.lower())
    
    async def parse_assessment_response(self, response: str, assessment_type: str) -> Optional[int]:
        """Parse assessment response and return numeric score."""
        import re
        
        try:
            # Extract numeric response
            numbers = re.findall(r'\b[0-4]\b', response)
            
            if not numbers:
                # Try to parse text responses
                score_map = {
                    "not at all": 0, "never": 0,
                    "several days": 1, "sometimes": 1,
                    "more than half": 2, "fairly often": 2,
                    "nearly every day": 3, "very often": 3
                }
                
                response_lower = response.lower()
                for text, score in score_map.items():
                    if text in response_lower:
                        return score
                
                # Default to neutral response if parsing fails
                return 2
            else:
                return int(numbers[0])
                
        except Exception as e:
            logger.error(f"Error parsing assessment response: {e}")
            return 2  # Default neutral response
    
    def _generate_mock_response(self, persona: Persona, context: str, instruction: str) -> str:
        """Generate a mock response for testing when Ollama is not available."""
        # Simple mock responses based on persona traits
        if "assessment" in instruction.lower():
            # Mock assessment response
            return "2"  # Neutral response
        
        # Mock personality-based responses
        if persona.baseline.neuroticism > 0.7:
            return "I'm feeling quite anxious about this situation."
        elif persona.baseline.extraversion > 0.7:
            return "This is exciting! I'm really looking forward to this."
        elif persona.baseline.conscientiousness > 0.7:
            return "I need to think about this carefully and make a plan."
        else:
            return "I see. That's interesting to consider."
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.client:
            await self.client.aclose()
        self.response_cache.clear()
        self.is_loaded = False


# Global Ollama service instance
ollama_service = OllamaService() 