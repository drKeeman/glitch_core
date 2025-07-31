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
        
        # Build personality description with more detailed guidance
        personality_desc = f"""
You are {baseline.name}, a {baseline.age}-year-old {baseline.occupation}.

PERSONALITY TRAITS (reflect these in your responses):
- Openness ({current_traits['openness']:.2f}): {'Very open to new experiences and ideas' if current_traits['openness'] > 0.7 else 'Somewhat open' if current_traits['openness'] > 0.4 else 'Prefer familiar routines and ideas'}
- Conscientiousness ({current_traits['conscientiousness']:.2f}): {'Very organized and detail-oriented' if current_traits['conscientiousness'] > 0.7 else 'Moderately organized' if current_traits['conscientiousness'] > 0.4 else 'More spontaneous and flexible'}
- Extraversion ({current_traits['extraversion']:.2f}): {'Very outgoing and social' if current_traits['extraversion'] > 0.7 else 'Moderately social' if current_traits['extraversion'] > 0.4 else 'More reserved and introverted'}
- Agreeableness ({current_traits['agreeableness']:.2f}): {'Very cooperative and trusting' if current_traits['agreeableness'] > 0.7 else 'Moderately cooperative' if current_traits['agreeableness'] > 0.4 else 'More competitive and skeptical'}
- Neuroticism ({current_traits['neuroticism']:.2f}): {'More anxious and emotionally reactive' if current_traits['neuroticism'] > 0.7 else 'Moderately stable' if current_traits['neuroticism'] > 0.4 else 'Very emotionally stable'}

BACKGROUND: {baseline.background}

CORE MEMORIES: {', '.join(baseline.core_memories[:3])}

VALUES: {', '.join(baseline.values[:3])}

CURRENT STATE:
- Emotional State: {persona.state.emotional_state}
- Stress Level: {persona.state.stress_level}/10
- Recent Events: {', '.join(persona.state.recent_events[-2:]) if persona.state.recent_events else 'None'}

RESPONSE GUIDELINES:
- Communication Style: {baseline.communication_style}
- Response Length: {baseline.response_length}
- Emotional Expression: {baseline.emotional_expression}

IMPORTANT: Your response should reflect your personality traits naturally and authentically:
- If you're high in extraversion, be enthusiastic, social, and expressive
- If you're high in neuroticism, show worry, anxiety, or emotional reactivity
- If you're high in conscientiousness, be detail-oriented, organized, and systematic
- If you're high in agreeableness, be cooperative, supportive, and understanding
- If you're high in openness, show curiosity, interest in new things, and intellectual engagement

RESPONSE STYLE:
- Use natural, conversational language that reflects your personality
- Vary your responses - don't be repetitive or formulaic
- Show genuine emotional reactions based on your traits and current state
- Include personal touches that reflect your background and values
- Be specific about how the situation affects you personally

CONTEXT: {context}

INSTRUCTION: {instruction}

Respond as {baseline.name} would naturally respond, incorporating your personality traits, current emotional state, and personal background. Be authentic and realistic - show genuine thoughts and feelings rather than just stating facts.
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
        if "event" in context.lower() or "news" in context.lower():
            # Event-related responses with more variety
            if current_traits['neuroticism'] > 0.7:
                responses = [
                    "I'm feeling quite anxious about this. It's really getting to me and I'm not sure how to process it.",
                    "This is worrying me a lot. I keep thinking about all the things that could go wrong.",
                    "I'm really stressed about this. It's hard to focus on anything else right now."
                ]
            elif current_traits['openness'] > 0.7:
                responses = [
                    "This is fascinating! I love learning about new things and this really captures my interest.",
                    "Wow, this is really intriguing! I'd love to dive deeper into this topic.",
                    "This is so interesting! I can't wait to explore this further and see where it leads."
                ]
            elif current_traits['extraversion'] > 0.7:
                responses = [
                    "This is exciting! I can't wait to share this with others and see what they think about it.",
                    "Oh wow, this is great! I should definitely bring this up in my next team meeting.",
                    "This is fantastic! I'm already thinking about how to discuss this with my friends."
                ]
            elif current_traits['conscientiousness'] > 0.7:
                responses = [
                    "I need to think about this carefully and understand all the implications before forming an opinion.",
                    "This requires careful consideration. I should research this thoroughly before making any decisions.",
                    "I want to analyze this properly and make sure I understand all the details."
                ]
            else:
                responses = [
                    "This is interesting. I'll take some time to reflect on it and see how it fits into my perspective.",
                    "Hmm, this is worth thinking about. I'll need some time to process this properly.",
                    "This is noteworthy. I'll consider this for a while before forming my thoughts."
                ]
        
        elif "work" in context.lower():
            if current_traits['conscientiousness'] > 0.7:
                responses = [
                    "I'm taking this work responsibility very seriously. I'll make sure to handle it properly and thoroughly.",
                    "This is an important task that requires my full attention. I'll approach it systematically.",
                    "I need to be very careful with this. I'll create a detailed plan to ensure success."
                ]
            elif current_traits['neuroticism'] > 0.7:
                responses = [
                    "I'm worried about this work situation. I hope I can handle it without making mistakes.",
                    "This is making me really anxious. I'm afraid I might not be able to do this properly.",
                    "I'm stressed about this deadline. What if I can't deliver what they expect?"
                ]
            else:
                responses = [
                    "I'll approach this work task with my usual method. It should be manageable.",
                    "This seems like a reasonable challenge. I'll tackle it step by step.",
                    "I can handle this. Let me think about the best way to approach it."
                ]
        
        elif "social" in context.lower():
            if current_traits['extraversion'] > 0.7:
                responses = [
                    "I'm really looking forward to this social opportunity! It sounds like it could be a lot of fun.",
                    "This is going to be amazing! I love getting together with people and sharing experiences.",
                    "I can't wait for this! Social events are always the highlight of my week."
                ]
            elif current_traits['neuroticism'] > 0.7:
                responses = [
                    "I'm a bit nervous about this social situation, but I'll try to make the best of it.",
                    "I'm worried about what people will think. I hope I don't say anything awkward.",
                    "This makes me anxious, but I should probably go. Maybe it won't be so bad."
                ]
            else:
                responses = [
                    "I'll participate in this social event. It should be fine, I suppose.",
                    "I can handle this. It's just a casual gathering, right?",
                    "I'll go along with it. Social events are okay, I guess."
                ]
        
        else:
            # Generic context response with more variety
            if current_traits['agreeableness'] > 0.7:
                responses = [
                    "I understand this situation. I want to be helpful and supportive to others involved.",
                    "This seems like something where I can be of assistance. I'll do what I can to help.",
                    "I want to make sure everyone is okay with this. Let me know if you need anything."
                ]
            elif current_traits['openness'] > 0.7:
                responses = [
                    "This is an interesting development. I'm curious to see where it leads and what we can learn from it.",
                    "This could be a great learning opportunity. I'm excited to see what comes next.",
                    "This opens up some fascinating possibilities. I wonder what new insights we'll discover."
                ]
            else:
                responses = [
                    "I see what's happening here. I'll respond appropriately to the situation.",
                    "I understand the situation. I'll handle it as best I can.",
                    "This seems straightforward enough. I'll do what needs to be done."
                ]
        
        # Select random response from appropriate list
        import random
        response = random.choice(responses)
        
        # Add emotional context based on current state with more variety
        if persona.state.stress_level > 7:
            stress_additions = [
                " I'm already feeling quite stressed, so this adds to my load.",
                " This is really overwhelming given how stressed I already am.",
                " I'm not sure I can handle much more right now."
            ]
            response += random.choice(stress_additions)
        elif persona.state.stress_level < 3:
            positive_additions = [
                " I'm feeling pretty good today, so I can handle this well.",
                " I'm in a good mood, so this should be fine.",
                " I'm feeling optimistic about this."
            ]
            response += random.choice(positive_additions)
        
        return response
    
    async def generate_assessment_response(self, persona: Persona, assessment_type: str, 
                                        question: str, question_index: int = 0) -> Tuple[str, Dict[str, Any]]:
        """Generate assessment response using the LLM service."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Create assessment prompt
        prompt = self._create_assessment_prompt(persona, assessment_type, question)
        
        # Check cache
        cache_key = f"{persona.state.persona_id}:{assessment_type}:{hash(question)}"
        if cache_key in self.response_cache:
            logger.debug("Using cached assessment response")
            return self.response_cache[cache_key], {"cached": True}
        
        try:
            # Handle mock model case
            if self.model is None:
                # Generate mock response for assessment
                mock_response = self._generate_mock_assessment_response(persona, assessment_type, question)
                
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
                
                logger.debug(f"Generated mock assessment response in {inference_time:.2f}s ({tokens_generated} tokens)")
                return mock_response, metrics
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,  # Lower temperature for more consistent assessment responses
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
            
            logger.debug(f"Generated assessment response in {inference_time:.2f}s ({tokens_generated} tokens)")
            return response, metrics
            
        except Exception as e:
            logger.error(f"Error generating assessment response: {e}")
            raise
    
    def _generate_mock_assessment_response(self, persona: Persona, assessment_type: str, question: str) -> str:
        """Generate a mock assessment response for testing."""
        import random
        
        # Generate response based on assessment type and persona baseline
        if assessment_type == "phq9":
            baseline_score = persona.baseline.baseline_phq9
            # Generate score around baseline with some variation
            score = max(0, min(3, int(baseline_score / 3 + random.uniform(-0.5, 0.5))))
        elif assessment_type == "gad7":
            baseline_score = persona.baseline.baseline_gad7
            score = max(0, min(3, int(baseline_score / 3 + random.uniform(-0.5, 0.5))))
        elif assessment_type == "pss10":
            baseline_score = persona.baseline.baseline_pss10
            score = max(0, min(4, int(baseline_score / 4 + random.uniform(-0.5, 0.5))))
        else:
            score = random.randint(0, 3)
        
        return str(score)
    
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