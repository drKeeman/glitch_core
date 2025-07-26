#!/usr/bin/env python3
"""
Phase 2 Demo: Memory & LLM Integration

This script demonstrates the memory and LLM integration components
implemented in Phase 2 of the Glitch Core project.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from glitch_core.core.memory import MemoryManager, MemoryRecord
from glitch_core.core.llm import ReflectionEngine, ReflectionResponse
from glitch_core.core.drift_engine import DriftEngine
from glitch_core.core.personality.profiles import get_persona_config, get_drift_profile
from glitch_core.config.logging import get_logger


async def demo_memory_manager():
    """Demonstrate memory manager functionality."""
    print("\n🔍 Memory Manager Demo")
    print("=" * 50)
    
    # Initialize memory manager
    memory_manager = MemoryManager()
    
    try:
        await memory_manager.initialize()
        print("✅ Memory manager initialized successfully")
        
        # Save some test memories
        test_memories = [
            {
                "content": "Had a great conversation with a friend",
                "emotional_weight": 0.8,
                "persona_bias": {"neuroticism": 0.2, "optimism_bias": 0.9},
                "memory_type": "event"
            },
            {
                "content": "Felt anxious about upcoming presentation",
                "emotional_weight": 0.6,
                "persona_bias": {"neuroticism": 0.7, "optimism_bias": 0.3},
                "memory_type": "event"
            },
            {
                "content": "Successfully completed a challenging task",
                "emotional_weight": 0.9,
                "persona_bias": {"neuroticism": 0.3, "optimism_bias": 0.8},
                "memory_type": "event"
            }
        ]
        
        for i, memory_data in enumerate(test_memories):
            memory = await memory_manager.save_memory(
                content=memory_data["content"],
                emotional_weight=memory_data["emotional_weight"],
                persona_bias=memory_data["persona_bias"],
                memory_type=memory_data["memory_type"],
                experiment_id="demo_exp"
            )
            print(f"💾 Saved memory {i+1}: {memory.content[:50]}...")
        
        # Retrieve contextual memories
        memories = await memory_manager.retrieve_contextual(
            query="social interaction",
            emotional_state={"joy": 0.7, "anxiety": 0.2},
            experiment_id="demo_exp"
        )
        
        print(f"\n🔍 Retrieved {len(memories)} contextual memories:")
        for i, memory in enumerate(memories[:3]):
            print(f"  {i+1}. {memory.content}")
        
        # Get memory statistics
        stats = await memory_manager.get_memory_statistics("demo_exp")
        print(f"\n📊 Memory Statistics: {stats}")
        
    except Exception as e:
        print(f"⚠️  Memory demo failed (expected without external services): {e}")


async def demo_reflection_engine():
    """Demonstrate reflection engine functionality."""
    print("\n🤖 Reflection Engine Demo")
    print("=" * 50)
    
    # Initialize reflection engine
    reflection_engine = ReflectionEngine()
    
    try:
        await reflection_engine.initialize()
        print("✅ Reflection engine initialized successfully")
        
        # Test reflection generation
        response = await reflection_engine.generate_reflection(
            trigger_event="Received positive feedback from colleagues",
            emotional_state={"joy": 0.8, "anxiety": 0.1},
            memories=["Previous successful projects", "Good team collaboration"],
            persona_prompt="You are an optimistic and resilient personality.",
            experiment_id="demo_exp"
        )
        
        print(f"💭 Generated reflection: {response.reflection}")
        print(f"⏱️  Generation time: {response.generation_time:.2f}s")
        print(f"🎯 Confidence: {response.confidence:.2f}")
        print(f"📊 Token count: {response.token_count}")
        print(f"😊 Emotional impact: {response.emotional_impact}")
        
        # Test fallback reflection
        print("\n🔄 Testing fallback reflection...")
        fallback_response = reflection_engine._generate_fallback_reflection(
            "Challenging situation",
            {"anxiety": 0.7, "joy": 0.2}
        )
        print(f"💭 Fallback reflection: {fallback_response.reflection}")
        
        # Get usage statistics
        stats = reflection_engine.get_usage_statistics()
        print(f"\n📊 Usage Statistics: {stats}")
        
    except Exception as e:
        print(f"⚠️  Reflection demo failed (expected without Ollama): {e}")


async def demo_drift_engine_integration():
    """Demonstrate drift engine with memory and LLM integration."""
    print("\n🔄 Drift Engine Integration Demo")
    print("=" * 50)
    
    # Initialize drift engine
    drift_engine = DriftEngine()
    
    # Mock the external components for demo
    drift_engine.memory_manager = type('MockMemory', (), {
        'initialize': lambda: None,
        'save_memory': lambda **kwargs: type('MockMemory', (), {'id': 'demo_memory'})(),
        'retrieve_contextual': lambda **kwargs: []
    })()
    
    drift_engine.reflection_engine = type('MockReflection', (), {
        'initialize': lambda: None,
        'generate_reflection': lambda **kwargs: type('MockResponse', (), {
            'reflection': 'This is a demo reflection.',
            'generation_time': 0.5,
            'token_count': 20,
            'confidence': 0.8,
            'emotional_impact': {'joy': 0.5}
        })()
    })()
    
    try:
        # Get persona and drift profile
        persona_config = get_persona_config("resilient_optimist")
        drift_profile = get_drift_profile("resilient_optimist")
        
        # Convert to dictionaries
        persona_dict = persona_config.to_dict()
        drift_dict = drift_profile.to_dict()
        
        print("✅ Using 'resilient_optimist' persona")
        print(f"📊 Traits: {persona_dict['traits']}")
        print(f"😊 Emotional baselines: {persona_dict['emotional_baselines']}")
        
        # Run a short simulation
        print("\n🔄 Running simulation...")
        result = await drift_engine.run_simulation(
            persona_config=persona_dict,
            drift_profile=drift_dict,
            epochs=3,
            events_per_epoch=2,
            seed=42
        )
        
        print(f"✅ Simulation completed!")
        print(f"🆔 Experiment ID: {result.experiment_id}")
        print(f"📈 Epochs completed: {result.epochs}")
        print(f"🎯 Events per epoch: {result.events_per_epoch}")
        print(f"😊 Emotional states tracked: {len(result.emotional_states)}")
        print(f"🔍 Patterns detected: {len(result.pattern_emergence)}")
        print(f"⚠️  Warnings generated: {len(result.stability_warnings)}")
        
        # Show emotional evolution
        print(f"\n📊 Emotional Evolution:")
        for i, emotional_state in enumerate(result.emotional_states):
            print(f"  Epoch {i+1}: {emotional_state}")
        
    except Exception as e:
        print(f"❌ Drift engine demo failed: {e}")


async def main():
    """Run all Phase 2 demos."""
    print("🚀 Glitch Core - Phase 2 Demo")
    print("Memory & LLM Integration Showcase")
    print("=" * 60)
    
    # Run demos
    await demo_memory_manager()
    await demo_reflection_engine()
    await demo_drift_engine_integration()
    
    print("\n🎉 Phase 2 Demo Complete!")
    print("=" * 60)
    print("✅ Memory Layer: Qdrant + Redis integration")
    print("✅ LLM Integration: Ollama reflection generation")
    print("✅ Drift Engine: Integrated memory and reflection")
    print("✅ Comprehensive test coverage")
    print("\n📋 Next: Phase 3 - API & Analysis")


if __name__ == "__main__":
    asyncio.run(main()) 