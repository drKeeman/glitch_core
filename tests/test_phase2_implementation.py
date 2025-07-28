"""
Comprehensive tests for Phase 2 implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any

from glitch_core.core.memory import (
    MemoryManager, MemoryRecord, MemoryCompressor, CompressedMemory,
    TemporalDecay, DecayConfig, RelevanceScorer, RelevanceConfig, RelevanceScore,
    MemoryVisualizer, MemoryVisualizationData
)
from glitch_core.core.analysis import (
    DriftAnalyzer, DriftPattern, StabilityAnalysis, PersonalityEvolution
)
from glitch_core.core.interventions import (
    InterventionFramework, Intervention, InterventionImpact,
    InterventionTemplate, InterventionType, InterventionIntensity
)


class TestPhase2MemoryOptimization:
    """Test Phase 2.1: Memory Management Optimization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_records = self._create_sample_memories()
        self.compressor = MemoryCompressor()
        self.temporal_decay = TemporalDecay()
        self.relevance_scorer = RelevanceScorer()
        self.visualizer = MemoryVisualizer()
    
    def _create_sample_memories(self) -> List[MemoryRecord]:
        """Create sample memory records for testing."""
        memories = []
        base_time = datetime.utcnow()
        
        for i in range(50):
            memory = MemoryRecord(
                id=f"memory_{i}",
                content=f"Sample memory content {i}",
                emotional_weight=0.3 + (i % 3) * 0.2,
                persona_bias={
                    "neuroticism": 0.4 + (i % 5) * 0.1,
                    "extraversion": 0.5 + (i % 3) * 0.15,
                    "conscientiousness": 0.6 + (i % 4) * 0.1
                },
                timestamp=base_time + timedelta(hours=i),
                memory_type=["event", "reflection", "intervention"][i % 3],
                context={"source": f"test_{i}", "significance": "medium"},
                decay_rate=0.1
            )
            memories.append(memory)
        
        return memories
    
    def test_memory_compression(self):
        """Test memory compression functionality."""
        # Test compression
        compressed_memories = self.compressor.compress_memories(
            self.memory_records, compression_threshold=0.8
        )
        
        assert len(compressed_memories) > 0
        assert len(compressed_memories) < len(self.memory_records)
        
        # Test compression ratio - be more flexible with expectations
        compression_ratio = len(compressed_memories) / len(self.memory_records)
        assert 0.01 <= compression_ratio <= 0.9  # Allow for very high compression
        
        # Test compressed memory structure
        for compressed in compressed_memories:
            assert isinstance(compressed, CompressedMemory)
            assert compressed.memory_count > 1
            assert compressed.compression_ratio < 1.0
            assert len(compressed.original_memory_ids) > 1
    
    def test_temporal_decay(self):
        """Test temporal decay calculations."""
        current_time = datetime.utcnow()
        
        # Test decay calculation for single memory
        memory = self.memory_records[0]
        decay_strength = self.temporal_decay.calculate_decay(
            memory, current_time
        )
        
        assert 0.0 <= decay_strength <= 1.0
        
        # Test batch decay calculation
        decay_strengths = self.temporal_decay.calculate_decay_batch(
            self.memory_records, current_time
        )
        
        assert len(decay_strengths) == len(self.memory_records)
        for memory_id, strength in decay_strengths.items():
            assert 0.0 <= strength <= 1.0
        
        # Test forgotten memories detection
        forgotten_ids = self.temporal_decay.find_forgotten_memories(
            self.memory_records, current_time
        )
        
        assert isinstance(forgotten_ids, list)
    
    def test_relevance_scoring(self):
        """Test relevance scoring functionality."""
        query = "test query"
        emotional_state = {"joy": 0.6, "anxiety": 0.2}
        current_time = datetime.utcnow()
        
        # Test relevance scoring
        relevance_scores = self.relevance_scorer.score_memories(
            self.memory_records, query, emotional_state, current_time
        )
        
        assert len(relevance_scores) == len(self.memory_records)
        
        # Test that scores are sorted by relevance
        for i in range(1, len(relevance_scores)):
            assert relevance_scores[i-1].total_score >= relevance_scores[i].total_score
        
        # Test top relevant memories
        top_memories = self.relevance_scorer.get_top_relevant_memories(
            self.memory_records, query, emotional_state, limit=5
        )
        
        assert len(top_memories) <= 5
        assert all(isinstance(m, MemoryRecord) for m in top_memories)
    
    def test_memory_visualization(self):
        """Test memory visualization functionality."""
        # Test timeline creation
        timeline_fig = self.visualizer.create_memory_timeline(self.memory_records)
        assert timeline_fig is not None
        
        # Test emotional distribution
        emotional_fig = self.visualizer.create_emotional_distribution(self.memory_records)
        assert emotional_fig is not None
        
        # Test decay analysis
        current_time = datetime.utcnow()
        decay_fig = self.visualizer.create_decay_analysis(
            self.memory_records, current_time, self.temporal_decay
        )
        assert decay_fig is not None
        
        # Test summary report
        viz_data = MemoryVisualizationData(
            memories=self.memory_records,
            compressed_memories=[],
            relevance_scores=[],
            decay_strengths={},
            time_range=(datetime.utcnow(), datetime.utcnow()),
            memory_types=["event", "reflection"],
            emotional_dimensions=["joy", "anxiety"]
        )
        
        summary = self.visualizer.create_memory_summary_report(viz_data)
        assert isinstance(summary, dict)
        assert "total_memories" in summary


class TestPhase2AnalysisEnhancement:
    """Test Phase 2.2: Analysis Capabilities Enhancement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.drift_analyzer = DriftAnalyzer()
        self.emotional_states = self._create_sample_emotional_states()
        self.trait_evolution = self._create_sample_trait_evolution()
    
    def _create_sample_emotional_states(self) -> List[Dict[str, float]]:
        """Create sample emotional states for testing."""
        states = []
        for i in range(100):
            state = {
                "joy": 0.3 + 0.4 * (i % 10) / 10,
                "sadness": 0.2 + 0.3 * ((i + 5) % 10) / 10,
                "anger": 0.1 + 0.2 * ((i + 3) % 10) / 10,
                "fear": 0.1 + 0.2 * ((i + 7) % 10) / 10,
                "anxiety": 0.2 + 0.3 * ((i + 2) % 10) / 10
            }
            states.append(state)
        return states
    
    def _create_sample_trait_evolution(self) -> Dict[str, List[float]]:
        """Create sample trait evolution data."""
        evolution = {}
        traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        
        for trait in traits:
            values = []
            for i in range(100):
                # Create some trend and variation
                base_value = 0.5
                trend = 0.001 * i  # Small trend
                noise = 0.1 * (i % 10) / 10  # Some noise
                value = base_value + trend + noise
                values.append(max(0.0, min(1.0, value)))
            evolution[trait] = values
        
        return evolution
    
    def test_stability_analysis(self):
        """Test stability trend analysis."""
        analysis = self.drift_analyzer.analyze_stability_trends(self.emotional_states)
        
        assert isinstance(analysis, StabilityAnalysis)
        assert 0.0 <= analysis.overall_stability <= 1.0
        assert 0.0 <= analysis.emotional_volatility <= 1.0
        assert 0.0 <= analysis.breakdown_risk <= 1.0
        assert 0.0 <= analysis.resilience_score <= 1.0
        assert 0.0 <= analysis.adaptation_rate <= 1.0
        assert isinstance(analysis.stability_trends, list)
        assert isinstance(analysis.critical_points, list)
    
    def test_pattern_detection(self):
        """Test pattern emergence detection."""
        patterns = self.drift_analyzer.detect_pattern_emergence(self.emotional_states)
        
        assert isinstance(patterns, list)
        
        for pattern in patterns:
            assert isinstance(pattern, DriftPattern)
            assert pattern.pattern_type in ["oscillation", "increasing_trend", "decreasing_trend", "clustering", "sudden_changes"]
            assert 0.0 <= pattern.confidence <= 1.0
            assert 0.0 <= pattern.impact_score <= 1.0
            assert isinstance(pattern.characteristics, dict)
            assert isinstance(pattern.description, str)
    
    def test_personality_evolution_analysis(self):
        """Test personality evolution analysis."""
        evolution = self.drift_analyzer.analyze_personality_evolution(self.trait_evolution)
        
        assert isinstance(evolution, PersonalityEvolution)
        assert evolution.trait_evolution == self.trait_evolution
        assert isinstance(evolution.trait_correlations, dict)
        assert isinstance(evolution.evolution_clusters, list)
        assert isinstance(evolution.significant_changes, list)
        assert isinstance(evolution.evolution_summary, dict)
    
    def test_persona_comparison(self):
        """Test persona comparison functionality."""
        persona_data = {
            "persona_1": self.emotional_states[:50],
            "persona_2": self.emotional_states[50:]
        }
        
        comparison = self.drift_analyzer.compare_personas(persona_data)
        
        assert isinstance(comparison, dict)
        assert "persona_count" in comparison
        assert "comparisons" in comparison
        assert "similarity_matrix" in comparison
        assert comparison["persona_count"] == 2
    
    def test_export_functionality(self):
        """Test export functionality."""
        analysis_results = {
            "stability_analysis": {"overall_stability": 0.7},
            "patterns": [{"type": "oscillation", "confidence": 0.8}],
            "evolution": {"trait_count": 5}
        }
        
        # Test JSON export
        json_export = self.drift_analyzer.export_analysis_results(analysis_results, "json")
        assert isinstance(json_export, str)
        assert "overall_stability" in json_export


class TestPhase2InterventionFramework:
    """Test Phase 2.3: Intervention Framework Enhancement."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.intervention_framework = InterventionFramework()
        self.experiment_id = "test_experiment_123"
    
    def test_intervention_creation(self):
        """Test intervention creation."""
        intervention = self.intervention_framework.create_intervention(
            experiment_id=self.experiment_id,
            intervention_type=InterventionType.TRAUMA,
            intensity=0.7,
            description="Test trauma intervention",
            applied_at_epoch=50,
            target_traits=["neuroticism", "anxiety"],
            parameters={"severity": "high", "duration": 10}
        )
        
        assert isinstance(intervention, Intervention)
        assert intervention.experiment_id == self.experiment_id
        assert intervention.intervention_type == InterventionType.TRAUMA
        assert intervention.intensity == 0.7
        assert intervention.status == "applied"
        assert len(intervention.target_traits) == 2
    
    def test_intervention_templates(self):
        """Test intervention template functionality."""
        templates = self.intervention_framework.get_intervention_templates()
        
        assert isinstance(templates, dict)
        assert len(templates) > 0
        
        # Test template structure
        for template_name, template in templates.items():
            assert isinstance(template, InterventionTemplate)
            assert template.name == template_name
            assert isinstance(template.intervention_type, InterventionType)
            assert 0.0 <= template.default_intensity <= 1.0
            assert isinstance(template.target_traits, list)
            assert isinstance(template.expected_effects, dict)
    
    def test_impact_measurement(self):
        """Test intervention impact measurement."""
        # Create intervention
        intervention = self.intervention_framework.create_intervention(
            experiment_id=self.experiment_id,
            intervention_type=InterventionType.THERAPY,
            intensity=0.6,
            description="Test therapy intervention",
            applied_at_epoch=50
        )
        
        # Create pre and post data
        pre_data = {
            "emotional_state": {"anxiety": 0.8, "depression": 0.6},
            "stability_metrics": {"overall_stability": 0.4}
        }
        
        post_data = {
            "emotional_state": {"anxiety": 0.4, "depression": 0.3},
            "stability_metrics": {"overall_stability": 0.7}
        }
        
        # Measure impact
        impact = self.intervention_framework.measure_intervention_impact(
            intervention.id, pre_data, post_data, current_epoch=70
        )
        
        assert isinstance(impact, InterventionImpact)
        assert impact.intervention_id == intervention.id
        assert impact.experiment_id == self.experiment_id
        assert isinstance(impact.pre_intervention_metrics, dict)
        assert isinstance(impact.post_intervention_metrics, dict)
        assert isinstance(impact.impact_scores, dict)
        assert 0.0 <= impact.effectiveness_rating <= 1.0
    
    def test_intervention_history(self):
        """Test intervention history tracking."""
        # Create multiple interventions
        for i in range(3):
            self.intervention_framework.create_intervention(
                experiment_id=self.experiment_id,
                intervention_type=InterventionType.TRAUMA,
                intensity=0.5 + i * 0.1,
                description=f"Test intervention {i}",
                applied_at_epoch=50 + i * 10
            )
        
        # Get history
        history = self.intervention_framework.get_intervention_history(self.experiment_id)
        
        assert len(history) == 3
        assert all(isinstance(intervention, Intervention) for intervention in history)
        assert all(intervention.experiment_id == self.experiment_id for intervention in history)
    
    def test_intervention_comparison(self):
        """Test intervention comparison functionality."""
        # Create interventions
        intervention_ids = []
        for i in range(2):
            intervention = self.intervention_framework.create_intervention(
                experiment_id=self.experiment_id,
                intervention_type=InterventionType.TRAUMA,
                intensity=0.5 + i * 0.2,
                description=f"Test intervention {i}",
                applied_at_epoch=50 + i * 10
            )
            intervention_ids.append(intervention.id)
        
        # Add impact data
        for intervention_id in intervention_ids:
            pre_data = {"emotional_state": {"anxiety": 0.8}}
            post_data = {"emotional_state": {"anxiety": 0.4}}
            self.intervention_framework.measure_intervention_impact(
                intervention_id, pre_data, post_data, current_epoch=70
            )
        
        # Compare interventions
        comparison = self.intervention_framework.compare_interventions(intervention_ids)
        
        assert isinstance(comparison, dict)
        assert "intervention_count" in comparison
        assert "effectiveness_rankings" in comparison
        assert "type_analysis" in comparison
        assert comparison["intervention_count"] == 2
    
    def test_export_functionality(self):
        """Test intervention data export."""
        # Create intervention with impact
        intervention = self.intervention_framework.create_intervention(
            experiment_id=self.experiment_id,
            intervention_type=InterventionType.THERAPY,
            intensity=0.6,
            description="Test intervention",
            applied_at_epoch=50
        )
        
        # Add impact data
        pre_data = {"emotional_state": {"anxiety": 0.8}}
        post_data = {"emotional_state": {"anxiety": 0.4}}
        self.intervention_framework.measure_intervention_impact(
            intervention.id, pre_data, post_data, current_epoch=70
        )
        
        # Export data
        export_data = self.intervention_framework.export_intervention_data(
            self.experiment_id, format="json"
        )
        
        assert isinstance(export_data, str)
        assert "interventions" in export_data
        assert "impact_data" in export_data


class TestPhase2Integration:
    """Test Phase 2 integration across all components."""
    
    def test_memory_analysis_integration(self):
        """Test integration between memory and analysis components."""
        # Create memory records
        memory_records = []
        base_time = datetime.utcnow()
        
        for i in range(20):
            memory = MemoryRecord(
                id=f"memory_{i}",
                content=f"Memory {i}",
                emotional_weight=0.5,
                persona_bias={"neuroticism": 0.4, "extraversion": 0.6},
                timestamp=base_time + timedelta(hours=i),
                memory_type="event",
                context={"source": "test"},
                decay_rate=0.1
            )
            memory_records.append(memory)
        
        # Test memory compression
        compressor = MemoryCompressor()
        compressed = compressor.compress_memories(memory_records, 0.8)
        assert len(compressed) > 0
        
        # Test temporal decay
        decay = TemporalDecay()
        current_time = datetime.utcnow()
        decay_strengths = decay.calculate_decay_batch(memory_records, current_time)
        assert len(decay_strengths) == len(memory_records)
        
        # Test relevance scoring
        scorer = RelevanceScorer()
        relevance_scores = scorer.score_memories(
            memory_records, "test query", {"joy": 0.5}, current_time
        )
        assert len(relevance_scores) == len(memory_records)
    
    def test_analysis_intervention_integration(self):
        """Test integration between analysis and intervention components."""
        # Create emotional states
        emotional_states = []
        for i in range(50):
            state = {
                "joy": 0.3 + 0.4 * (i % 10) / 10,
                "anxiety": 0.2 + 0.3 * ((i + 5) % 10) / 10
            }
            emotional_states.append(state)
        
        # Analyze stability
        analyzer = DriftAnalyzer()
        stability_analysis = analyzer.analyze_stability_trends(emotional_states)
        assert isinstance(stability_analysis, StabilityAnalysis)
        
        # Create intervention based on analysis
        framework = InterventionFramework()
        intervention = framework.create_intervention(
            experiment_id="test_experiment",
            intervention_type=InterventionType.THERAPY,
            intensity=0.6,
            description="Intervention based on stability analysis",
            applied_at_epoch=25
        )
        assert isinstance(intervention, Intervention)
    
    def test_comprehensive_workflow(self):
        """Test comprehensive Phase 2 workflow."""
        # 1. Create memory records
        memory_records = []
        base_time = datetime.utcnow()
        for i in range(30):
            memory = MemoryRecord(
                id=f"memory_{i}",
                content=f"Memory {i}",
                emotional_weight=0.4 + (i % 3) * 0.2,
                persona_bias={"neuroticism": 0.3 + (i % 5) * 0.1},
                timestamp=base_time + timedelta(hours=i),
                memory_type=["event", "reflection"][i % 2],
                context={"source": "test"},
                decay_rate=0.1
            )
            memory_records.append(memory)
        
        # 2. Compress memories
        compressor = MemoryCompressor()
        compressed = compressor.compress_memories(memory_records, 0.8)
        assert len(compressed) > 0
        
        # 3. Analyze patterns
        analyzer = DriftAnalyzer()
        emotional_states = [{"joy": 0.5, "anxiety": 0.3} for _ in range(20)]
        patterns = analyzer.detect_pattern_emergence(emotional_states)
        assert isinstance(patterns, list)
        
        # 4. Create intervention
        framework = InterventionFramework()
        intervention = framework.create_intervention(
            experiment_id="test_experiment",
            intervention_type=InterventionType.THERAPY,
            intensity=0.6,
            description="Comprehensive test intervention",
            applied_at_epoch=10
        )
        assert isinstance(intervention, Intervention)
        
        # 5. Measure impact
        pre_data = {"emotional_state": {"anxiety": 0.8}}
        post_data = {"emotional_state": {"anxiety": 0.4}}
        impact = framework.measure_intervention_impact(
            intervention.id, pre_data, post_data, current_epoch=20
        )
        assert isinstance(impact, InterventionImpact)
        
        # 6. Export results
        export_data = framework.export_intervention_data("test_experiment", "json")
        assert isinstance(export_data, str)
        assert "interventions" in export_data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 