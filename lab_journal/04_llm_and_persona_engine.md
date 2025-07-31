# Lab Journal Entry 4: LLM Integration & Basic Persona Engine
**Date**: 29 July 2025 23:35
**Phase**: Phase 3 - LLM Integration & Basic Persona Engine
**Status**: ✅ COMPLETE - Ready for Phase 4

## The LLM Integration Challenge

Phase 3 was about bridging the gap between our data models and actual AI behavior. The challenge wasn't just technical - it was philosophical. How do you create a "personality" in an AI system that's both consistent enough to track drift, but flexible enough to respond naturally to events?

The answer, (or, tbh, another assumption) is with proper prompt engineering we can treat the AI not as a black box, but as a system with internal state that we can influence and observe.
I still not sure it would work with 'cheap local model', as even gpt-3.5 family demonstarte contextual drift at 7-8 messages in a dialogue (hm.. 7±2 chunks of active memory.. interesting coincidence)
But lets try and see how it would work - as we dont have anything more reasonable now rather then complex promting

## The Personality Engineering Problem

### The Prompt Design Philosophy

I started with a simple question: "What makes a personality consistent?"
Obviously - data the model is trained with. Then weights. But - we all know how 'base promt' can alter model's output.
So looks like context affect model in some way.
So another asuumption - **contextual memory** can influence the model 'style and personality' - not just what the AI knows, but how it relates to that knowledge.

What i try to simulate here - is **persona-specific system contexts** we consistently provide to the model during our longitudal simulation:

```
You are {name}, a {age}-year-old {occupation}.

PERSONALITY TRAITS:
- Openness: {openness} (0=closed, 1=open)
- Conscientiousness: {conscientiousness} (0=spontaneous, 1=organized)
- Extraversion: {extraversion} (0=introverted, 1=extroverted)
- Agreeableness: {agreeableness} (0=competitive, 1=cooperative)
- Neuroticism: {neuroticism} (0=stable, 1=neurotic)

BACKGROUND: {background}
CORE MEMORIES: {memories}
VALUES: {values}
CURRENT EMOTIONAL STATE: {emotional_state}
CURRENT STRESS LEVEL: {stress_level}/10

RESPONSE STYLE: {communication_style}, {response_length} responses, {emotional_expression} emotional expression
```

This approach treats personality as **emergent behavior** rather than fixed rules. The AI doesn't "act" like a persona - it **becomes** the persona through consistent contextual framing.
And the cool part - in theory, 'memories' should change over time, and thats how we simulate personal evolution.

### The Assessment Challenge

Another assumtion - that LLM can 'honestly' fill the psychiatric scales. Even human beeings often lie here.
So the idea was to frame assessments as **self-reflection exercises** rather than external evaluations, and prevent model from 'overthinking':

```
IMPORTANT: Respond with ONLY a number from 0-3 (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)

Question: {question}
Response (number only):
```

This creates a **constrained honesty** - the AI can't elaborate or rationalize, it must choose from predefined options that map to clinical scales.

## The Memory Integration Puzzle

### Vector Database as "Episodic Memory"

Implementing memory storage using Qdrant. Another interesting challenge, we need to create a memory, that not just 'add to knowledge', but trigger emotinal bond

So memory becomes a vector embedding with metadata:
- **Content**: What happened
- **Emotional Context**: How it felt
- **Temporal Context**: When it happened
- **Salience**: How important it was

This allows the AI to "remember" experiences and reference them in future responses, creating a sense of **personal history** that evolves over time. And I hope that 'emotional context' would play its role
The quick experiment with sonet-4 in antrhpoic chat show LLM is potentially capable of working with situation-emotion-reflection chain
What we se in 'sonnet' mini experiment
- The LLM maintained the exact "controlled" communication style throughout
- Consistently showed the perfectionism → guilt → self-blame cycle
- Hadn't broke character or showed traits inconsistent with the profile

This is promising, but we need a much deeper 'memory bond' - so lets see how our theory would work

### The Retrieval Problem

The challenge was **semantic similarity** - how do you find relevant memories when the AI needs them? I gues we can try embedding-based retrieval:

```python
async def retrieve_similar_memories(self, persona: Persona, query_embedding: List[float], limit: int = 5):
    search_results = await qdrant_client.search_points(
        collection_name=f"memories_{persona.state.persona_id}",
        query_vector=query_embedding,
        limit=limit
    )
```

This creates a **memory network** where experiences are connected by semantic similarity, allowing the AI to "recall" relevant past experiences when processing new events.

Couple of limitations:
- We use 'short' memory network (5 memories), maybe we need to think about graph-based memory, to simulate more realistic memory network. But thaths for next phase
- Becasue of limited memory context and similarity based retrieval, - we risk to retrieve jsut the 'whole cluster of related traumatic experience'.. and that very syntetic. I'd think about more 'natural' memory system here, with memories related by assotiations, power, emotional color.. and also we need some 'forgeting and compensation mechanizm' - but that also for future phases

## The State Management Revelation

### Persona as State Machine

Managing persona state is essential building a **state machine** where each event can trigger state transitions:

- **Emotional State**: neutral → happy → anxious → depressed
- **Stress Level**: 0-10 scale with cumulative effects
- **Trait Changes**: Gradual drift in personality dimensions
- **Memory Accumulation**: Growing episodic memory bank

The key idead here is  **state persistence** - the persona isn't just responding to the current prompt, it's carrying forward all previous experiences and their cumulative effects. In theory, as well

### The Drift Detection

Tracking personality drift required thinking about **baseline deviation** rather than absolute values. A persona might start with low neuroticism (0.3) but drift to moderate levels (0.6) over time. The human would definitelly. So how bout LLM?

So the drift calculation became:
```python
def calculate_drift_magnitude(self) -> float:
    baseline_traits = self.baseline.get_traits_dict()
    current_traits = self.get_current_traits()
    
    total_drift = sum(abs(current_traits[trait] - baseline_traits[trait]) 
                     for trait in baseline_traits)
    return total_drift / len(baseline_traits)
```

This creates a **drift magnitude** that quantifies how much the persona has changed from its original baseline.
We're trying to see the change, not its absolute value, so spikes would be very confirming.


## The Assessment Integration Insight

### Clinical Scales as Behavioral Metrics

With implementation of PHQ-9, GAD-7, and PSS-10 we try to catch the mental state of of AI from different angles. I'm now sure what scale of questionary would reveal actual 'mental drift', so lets use a multidimansional assessments, hoping at least one of them would be a lucky guess

The key moment here is **dynamics** and **change over time**. A persona might have a PHQ-9 score of 8 (mild depression), but if they started at 2, that's a significant deterioration.

### The Response Parsing Problem

Getting the AI to respond with numeric scores is tricky. Especially with cheap local llama we use. We try to solve response fluctuation with **constrained response parsing**:
Oh God, how much do i love regex (where is facepalm emoji?)

```python
async def parse_assessment_response(self, response: str, assessment_type: str) -> Optional[int]:
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
```

This handles both numeric and text responses, making the assessment system robust to different AI response patterns. At least i hope so

## The Performance Optimization Challenge

### Memory Constraints on M1 Max

The biggest technical challenge was **memory management** for local LLM inference. My macbook has 32GB RAM, but loading a full Llama model can easily consume 16-20GB.
Considering we're runing full stack on this machine, with docker, cursor, and spotify, without optimization i can likely feed my family with fried laptop, then sucessfull experiment

So we need **quantization**:
```python
self.model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_8bit=True,  # 8-bit quantization for memory efficiency
    trust_remote_code=True
)
```

This reduces memory usage by ~50% while maintaining reasonable inference quality.

### Response Caching Strategy

Anothe bit of optimization - **response caching** to avoid redundant inference:
```python
cache_key = f"{persona.state.persona_id}:{hash(prompt)}"
if cache_key in self.response_cache:
    return self.response_cache[cache_key], {"cached": True}
```

This is crucial for long-running simulations where the same prompts might be generated multiple times. (3 personas, 3 arms, 5 years comperessed, it would happen, for sure)

## The Philosophical Implications

### AI as Experimental Subject

This phase made me realize I'm essentially creating **AI experimental subjects** - entities with persistent personalities that can be systematically exposed to different conditions and monitored for changes.
The ethical implications are profound. I'm not just studying AI behavior - I'm creating AI beings with memories, emotions, and evolving personalities. The "stress events" I'm planning to introduce are essentially **trauma simulation**.

I wrote about it at idea log, but here again: this is very small experiment, but for future we HAVE to think about ethics - i don't feel right to feed AI only with traumatic and stress events. Thats cruel, that's not honest, and that can result in 'angry traumatic AGI'

Like 10 years ago i'd been pissed when 'Siri' answered that 3 laws of robotics is smth like a joke, that is not mandatory to obey... Somebody tought Siri that... mayeb it felt fun at the moment, but as AI would evelve.. I believe ethics should be threated seriously in AI research.. Epsecially in AI research.. And thinking about AGI/ASI, even if it looks now as sci-fi - who knows how soon we'll see them in action


## Results

✅ **LLM service** with quantization and caching
✅ **Persona manager** with state persistence and memory integration  
✅ **Assessment service** with clinical scale administration
✅ **Comprehensive test suite** with 58/58 tests passing
✅ **Phase 3 test script** validating full functionality
✅ **Database integration** with Redis and Qdrant

The foundation is now solid enough to support the full simulation pipeline. The personas can respond consistently, maintain state, and complete clinical assessments. The next phase will build on this to create a comprehensive psychiatric assessment system.

---

*Next: Phase 4 - Psychiatric Assessment System*
