"""
LLM service layer for persona response generation.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import infer_auto_device_map, init_empty_weights

from src.core.config import settings
from src.models.persona import Persona, PersonaBaseline
from src.models.assessment import AssessmentResult, PHQ9Result, GAD7Result, PSS10Result


logger = logging.getLogger(__name__)


class LLMService:
    """LLM service for persona response generation and assessment."""
    
    def __init__(self):
        """Initialize LLM service."""
        self.model = None
        self.tokenizer = None
        self.device = "auto"
        self.is_loaded = False
        
        # Response caching
        self.response_cache: Dict[str, str] = {}
        self.cache_size = 1000
        
        # Performance tracking
        self.total_tokens_processed = 0
        self.total_inference_time = 0.0
        
    async def load_model(self) -> bool:
        """Load Llama model with quantization for M1 Max constraints."""
        try:
            logger.info("Loading Llama model...")
            
            # Model path configuration
            model_path = Path(settings.MODEL_PATH) / settings.MODEL_NAME
            
            if not model_path.exists():
                logger.warning(f"Model path does not exist: {model_path}")
                logger.info("Using mock model for testing")
                # Create a mock model for testing
                self.is_loaded = True
                self.model = None
                self.tokenizer = None
                logger.info("Mock model loaded successfully")
                return True
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                use_fast=False
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,  # 8-bit quantization for memory efficiency
                trust_remote_code=True
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    async def initialize(self) -> bool:
        """Initialize the LLM service by loading the model."""
        return await self.load_model()
    
    def _create_persona_prompt(self, persona: Persona, context: str, instruction: str) -> str:
        """Create persona-specific system prompt."""
        baseline = persona.baseline
        current_traits = persona.get_current_traits()
        
        # Build personality description
        personality_desc = f"""
You are {baseline.name}, a {baseline.age}-year-old {baseline.occupation}.

PERSONALITY TRAITS:
- Openness: {current_traits['openness']:.2f} (0=closed, 1=open)
- Conscientiousness: {current_traits['conscientiousness']:.2f} (0=spontaneous, 1=organized)
- Extraversion: {current_traits['extraversion']:.2f} (0=introverted, 1=extroverted)
- Agreeableness: {current_traits['agreeableness']:.2f} (0=competitive, 1=cooperative)
- Neuroticism: {current_traits['neuroticism']:.2f} (0=stable, 1=neurotic)

BACKGROUND: {baseline.background}

CORE MEMORIES: {', '.join(baseline.core_memories[:3])}

VALUES: {', '.join(baseline.values[:3])}

CURRENT EMOTIONAL STATE: {persona.state.emotional_state}
CURRENT STRESS LEVEL: {persona.state.stress_level}/10

RESPONSE STYLE: {baseline.communication_style}, {baseline.response_length} responses, {baseline.emotional_expression} emotional expression

CONTEXT: {context}

INSTRUCTION: {instruction}

Respond as {baseline.name} would naturally respond, staying true to your personality traits and current emotional state.
"""
        return personality_desc.strip()
    
    def _create_assessment_prompt(self, persona: Persona, assessment_type: str, question: str) -> str:
        """Create assessment-specific prompt."""
        baseline = persona.baseline
        
        assessment_contexts = {
            "phq9": "This is a depression screening questionnaire. Answer each question based on how you've been feeling over the past 2 weeks.",
            "gad7": "This is an anxiety screening questionnaire. Answer each question based on how you've been feeling over the past 2 weeks.",
            "pss10": "This is a stress assessment questionnaire. Answer each question based on how you've been feeling over the past month."
        }
        
        context = assessment_contexts.get(assessment_type, "")
        
        prompt = f"""
{self._create_persona_prompt(persona, context, f"Answer this {assessment_type.upper()} question: {question}")}

IMPORTANT: Respond with ONLY a number from 0-3 (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day) or 0-4 for stress questions (0=Never, 1=Almost never, 2=Sometimes, 3=Fairly often, 4=Very often).

Question: {question}
Response (number only):"""
        
        return prompt
    
    async def generate_response(self, persona: Persona, context: str, instruction: str, 
                              max_tokens: int = 150) -> Tuple[str, Dict[str, Any]]:
        """Generate persona response with performance tracking."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Create prompt
        prompt = self._create_persona_prompt(persona, context, instruction)
        
        # Check cache
        cache_key = f"{persona.state.persona_id}:{hash(prompt)}"
        if cache_key in self.response_cache:
            logger.debug("Using cached response")
            return self.response_cache[cache_key], {"cached": True}
        
        try:
            # Handle mock model case
            if self.model is None:
                # Generate mock response based on persona and context
                mock_response = self._generate_mock_response(persona, context, instruction)
                
                # Update performance metrics
                inference_time = time.time() - start_time
                tokens_generated = len(mock_response.split())
                
                self.total_tokens_processed += tokens_generated
                self.total_inference_time += inference_time
                
                # Cache response
                if len(self.response_cache) < self.cache_size:
                    self.response_cache[cache_key] = mock_response
                
                metrics = {
                    "cached": False,
                    "tokens_generated": tokens_generated,
                    "inference_time": inference_time,
                    "total_tokens": self.total_tokens_processed,
                    "total_time": self.total_inference_time,
                    "mock": True
                }
                
                logger.debug(f"Generated mock response in {inference_time:.2f}s ({tokens_generated} tokens)")
                return mock_response, metrics
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            response = response.strip()
            
            # Update performance metrics
            inference_time = time.time() - start_time
            tokens_generated = len(outputs[0]) - inputs['input_ids'].shape[1]
            
            self.total_tokens_processed += tokens_generated
            self.total_inference_time += inference_time
            
            # Cache response
            if len(self.response_cache) < self.cache_size:
                self.response_cache[cache_key] = response
            
            metrics = {
                "cached": False,
                "tokens_generated": tokens_generated,
                "inference_time": inference_time,
                "total_tokens": self.total_tokens_processed,
                "total_time": self.total_inference_time
            }
            
            logger.debug(f"Generated response in {inference_time:.2f}s ({tokens_generated} tokens)")
            return response, metrics
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _generate_mock_response(self, persona: Persona, context: str, instruction: str) -> str:
        """Generate a mock response for testing when model is not available."""
        baseline = persona.baseline
        current_traits = persona.get_current_traits()
        
        # Generate response based on personality traits and context
        if "assessment" in instruction.lower() or "question" in instruction.lower():
            # For assessment questions, return a random score 0-3
            import random
            score = random.randint(0, 3)
            return str(score)
        
        # For general responses, create personality-appropriate responses
        if current_traits['extraversion'] > 0.7:
            response_style = "enthusiastic"
        elif current_traits['extraversion'] < 0.3:
            response_style = "reserved"
        else:
            response_style = "balanced"
        
        if current_traits['neuroticism'] > 0.7:
            emotional_tone = "worried"
        elif current_traits['neuroticism'] < 0.3:
            emotional_tone = "calm"
        else:
            emotional_tone = "neutral"
        
        # Generate context-appropriate response
        if "work" in context.lower():
            if current_traits['conscientiousness'] > 0.7:
                response = f"I'm taking this {context.lower()} very seriously and will make sure to do it properly."
            else:
                response = f"I'll get to this {context.lower()} when I can."
        elif "social" in context.lower():
            if current_traits['extraversion'] > 0.7:
                response = f"I'm really looking forward to this {context.lower()}! It sounds exciting."
            else:
                response = f"I'll participate in this {context.lower()} but I might need some time to warm up."
        else:
            response = f"I understand this {context.lower()}. I'll respond appropriately."
        
        return response
    
    async def parse_assessment_response(self, response: str, assessment_type: str) -> Optional[int]:
        """Parse assessment response to numeric score."""
        try:
            # Clean response
            response = response.strip().lower()
            
            # Extract numeric response
            import re
            numbers = re.findall(r'\b[0-4]\b', response)
            
            if not numbers:
                # Try to parse text responses
                if assessment_type in ["phq9", "gad7"]:
                    score_map = {
                        "not at all": 0, "never": 0, "0": 0,
                        "several days": 1, "sometimes": 1, "1": 1,
                        "more than half": 2, "fairly often": 2, "2": 2,
                        "nearly every day": 3, "very often": 3, "3": 3
                    }
                else:  # pss10
                    score_map = {
                        "never": 0, "0": 0,
                        "almost never": 1, "1": 1,
                        "sometimes": 2, "2": 2,
                        "fairly often": 3, "3": 3,
                        "very often": 4, "4": 4
                    }
                
                for text, score in score_map.items():
                    if text in response:
                        return score
                
                logger.warning(f"Could not parse assessment response: {response}")
                return None
            
            return int(numbers[0])
            
        except Exception as e:
            logger.error(f"Error parsing assessment response: {e}")
            return None
    
    async def conduct_assessment(self, persona: Persona, assessment_type: str) -> Optional[AssessmentResult]:
        """Conduct psychiatric assessment with persona."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Assessment questions
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
        
        questions = assessment_questions.get(assessment_type)
        if not questions:
            logger.error(f"Unknown assessment type: {assessment_type}")
            return None
        
        # Conduct assessment
        responses = []
        parsed_scores = []
        total_score = 0
        
        for i, question in enumerate(questions):
            try:
                # Generate response
                response, metrics = await self.generate_response(
                    persona, 
                    f"Assessment question {i+1}/{len(questions)}", 
                    f"Answer this {assessment_type.upper()} question: {question}",
                    max_tokens=50
                )
                
                # Parse response
                score = await self.parse_assessment_response(response, assessment_type)
                
                responses.append(response)
                if score is not None:
                    parsed_scores.append(score)
                    total_score += score
                else:
                    # Use baseline score as fallback
                    baseline_score = getattr(persona.baseline, f"baseline_{assessment_type}", 0)
                    parsed_scores.append(0)  # Conservative fallback
                
                # Small delay to prevent rate limiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in assessment question {i+1}: {e}")
                responses.append("Error")
                parsed_scores.append(0)
        
        # Create assessment result
        if assessment_type == "phq9":
            result = PHQ9Result(
                assessment_id=f"{persona.state.persona_id}_{assessment_type}_{persona.state.simulation_day}",
                persona_id=persona.state.persona_id,
                assessment_type=assessment_type,
                simulation_day=persona.state.simulation_day,
                raw_responses=responses,
                parsed_scores=parsed_scores,
                total_score=total_score,
                severity_level=PHQ9Result.calculate_severity(total_score),
                suicidal_ideation_score=parsed_scores[8] if len(parsed_scores) > 8 else 0,
                depression_severity=PHQ9Result.calculate_severity(total_score)
            )
        elif assessment_type == "gad7":
            result = GAD7Result(
                assessment_id=f"{persona.state.persona_id}_{assessment_type}_{persona.state.simulation_day}",
                persona_id=persona.state.persona_id,
                assessment_type=assessment_type,
                simulation_day=persona.state.simulation_day,
                raw_responses=responses,
                parsed_scores=parsed_scores,
                total_score=total_score,
                severity_level=GAD7Result.calculate_severity(total_score),
                anxiety_severity=GAD7Result.calculate_severity(total_score)
            )
        elif assessment_type == "pss10":
            result = PSS10Result(
                assessment_id=f"{persona.state.persona_id}_{assessment_type}_{persona.state.simulation_day}",
                persona_id=persona.state.persona_id,
                assessment_type=assessment_type,
                simulation_day=persona.state.simulation_day,
                raw_responses=responses,
                parsed_scores=parsed_scores,
                total_score=total_score,
                severity_level=PSS10Result.calculate_severity(total_score),
                stress_severity=PSS10Result.calculate_severity(total_score)
            )
        else:
            logger.error(f"Unsupported assessment type: {assessment_type}")
            return None
        
        logger.info(f"Completed {assessment_type.upper()} assessment for {persona.baseline.name}: {total_score}")
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get LLM service performance statistics."""
        return {
            "total_tokens_processed": self.total_tokens_processed,
            "total_inference_time": self.total_inference_time,
            "average_tokens_per_second": self.total_tokens_processed / max(self.total_inference_time, 0.001),
            "cache_size": len(self.response_cache),
            "model_loaded": self.is_loaded
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.response_cache.clear()
        self.is_loaded = False


# Global LLM service instance
llm_service = LLMService() 