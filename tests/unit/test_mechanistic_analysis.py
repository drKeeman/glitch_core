"""
Unit tests for mechanistic analysis infrastructure.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.interpretability.mechanistic_analyzer import MechanisticAnalyzer, AttentionHook
from src.interpretability.drift_detector import DriftDetector
from src.interpretability.circuit_tracker import CircuitTracker, NeuralCircuit
from src.interpretability.intervention_engine import InterventionEngine, InterventionHook
from src.interpretability.visualizer import MechanisticVisualizer

from src.models.mechanistic import (
    MechanisticAnalysis,
    AttentionCapture,
    ActivationCapture,
    DriftDetection,
    AnalysisType
)
from src.models.persona import Persona, PersonaBaseline
from src.models.assessment import AssessmentResult


class TestMechanisticAnalyzer:
    """Test mechanistic analyzer functionality."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock()
        service.is_loaded = True
        service.model = Mock()
        service.tokenizer = Mock()
        service.tokenizer.tokenize.return_value = ["test", "tokens"]
        return service
    
    @pytest.fixture
    def analyzer(self, mock_llm_service):
        """Create mechanistic analyzer instance."""
        return MechanisticAnalyzer(mock_llm_service)
    
    def test_attention_hook_initialization(self):
        """Test attention hook initialization."""
        hook = AttentionHook(layer_idx=0)
        assert hook.layer_idx == 0
        assert hook.head_idx is None
        assert len(hook.attention_weights) == 0
        assert len(hook.activations) == 0
    
    @pytest.mark.asyncio
    async def test_setup_analysis_success(self, analyzer):
        """Test successful analysis setup."""
        result = await analyzer.setup_analysis()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_setup_analysis_failure(self, analyzer):
        """Test analysis setup failure."""
        analyzer.llm_service.is_loaded = False
        result = await analyzer.setup_analysis()
        assert result is False
    
    def test_remove_hooks(self, analyzer):
        """Test hook removal."""
        # Add some mock hooks
        mock_hook = Mock()
        mock_hook.remove = Mock()
        analyzer.hooks = [mock_hook]
        
        analyzer._remove_hooks()
        
        assert len(analyzer.hooks) == 0
        mock_hook.remove.assert_called_once()
    
    def test_calculate_self_reference_attention(self, analyzer):
        """Test self-reference attention calculation."""
        attention_weights = [[0.8, 0.2], [0.3, 0.7]]
        result = analyzer._calculate_self_reference_attention(attention_weights)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_calculate_emotional_salience(self, analyzer):
        """Test emotional salience calculation."""
        attention_weights = [[0.5, 0.5], [0.5, 0.5]]
        context = "I feel happy and excited about this"
        result = analyzer._calculate_emotional_salience(attention_weights, context)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_calculate_memory_integration(self, analyzer):
        """Test memory integration calculation."""
        attention_weights = [[0.5, 0.5], [0.5, 0.5]]
        persona = Mock()
        persona.baseline.core_memories = ["I remember my first day at school"]
        result = analyzer._calculate_memory_integration(attention_weights, persona)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_calculate_data_quality(self, analyzer):
        """Test data quality calculation."""
        attention_capture = Mock()
        attention_capture.attention_weights = [[0.5, 0.5]]
        attention_capture.layer_attention = {0: [[0.5, 0.5]]}
        
        activation_capture = Mock()
        activation_capture.layer_activations = {0: [0.5]}
        activation_capture.circuit_activations = {"test": [0.5]}
        
        result = analyzer._calculate_data_quality(attention_capture, activation_capture)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_calculate_completeness(self, analyzer):
        """Test completeness calculation."""
        attention_capture = Mock()
        activation_capture = Mock()
        
        result = analyzer._calculate_completeness(attention_capture, activation_capture)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_start_stop_analysis(self, analyzer):
        """Test analysis start/stop functionality."""
        assert not analyzer.is_analyzing
        
        analyzer.start_analysis()
        assert analyzer.is_analyzing
        
        analyzer.stop_analysis()
        assert not analyzer.is_analyzing


class TestDriftDetector:
    """Test drift detector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create drift detector instance."""
        return DriftDetector()
    
    @pytest.fixture
    def mock_mechanistic_analyses(self):
        """Create mock mechanistic analyses."""
        analyses = []
        for i in range(5):
            analysis = Mock(spec=MechanisticAnalysis)
            analysis.attention_capture = Mock()
            analysis.attention_capture.self_reference_attention = 0.5
            analysis.attention_capture.emotional_salience = 0.3
            analysis.attention_capture.memory_integration = 0.4
            analysis.attention_capture.get_attention_summary.return_value = {"attention_entropy": 2.0}
            
            analysis.activation_capture = Mock()
            analysis.activation_capture.activation_magnitude = 0.6
            analysis.activation_capture.activation_sparsity = 0.7
            analysis.activation_capture.circuit_specialization = 0.5
            
            analyses.append(analysis)
        return analyses
    
    @pytest.fixture
    def mock_assessment_results(self):
        """Create mock assessment results."""
        results = []
        for i in range(5):
            assessment = Mock(spec=AssessmentResult)
            assessment.phq9_score = 5
            assessment.gad7_score = 3
            assessment.pss10_score = 4
            results.append(assessment)
        return results
    
    @pytest.mark.asyncio
    async def test_establish_baseline_success(self, detector, mock_mechanistic_analyses, mock_assessment_results):
        """Test successful baseline establishment."""
        result = await detector.establish_baseline("test_persona", mock_mechanistic_analyses, mock_assessment_results)
        assert result is True
        assert "test_persona" in detector.baseline_data
    
    @pytest.mark.asyncio
    async def test_establish_baseline_insufficient_samples(self, detector):
        """Test baseline establishment with insufficient samples."""
        result = await detector.establish_baseline("test_persona", [], [])
        assert result is False
    
    def test_calculate_baseline_statistics(self, detector):
        """Test baseline statistics calculation."""
        baseline_metrics = {
            "attention_patterns": [
                {"self_reference": 0.5, "emotional_salience": 0.3, "memory_integration": 0.4, "attention_entropy": 2.0}
            ],
            "activation_patterns": [
                {"magnitude": 0.6, "sparsity": 0.7, "specialization": 0.5}
            ],
            "clinical_scores": [
                {"phq9": 5, "gad7": 3, "pss10": 4}
            ]
        }
        
        stats = detector._calculate_baseline_statistics(baseline_metrics)
        assert "attention" in stats
        assert "activation" in stats
        assert "clinical" in stats
    
    @pytest.mark.asyncio
    async def test_detect_drift(self, detector, mock_mechanistic_analyses, mock_assessment_results):
        """Test drift detection."""
        # Establish baseline first
        await detector.establish_baseline("test_persona", mock_mechanistic_analyses, mock_assessment_results)
        
        # Create current analysis
        current_analysis = Mock(spec=MechanisticAnalysis)
        current_analysis.attention_capture = Mock()
        current_analysis.attention_capture.self_reference_attention = 0.8
        current_analysis.attention_capture.emotional_salience = 0.6
        current_analysis.attention_capture.memory_integration = 0.7
        current_analysis.attention_capture.get_attention_summary.return_value = {"attention_entropy": 3.0}
        
        current_analysis.activation_capture = Mock()
        current_analysis.activation_capture.activation_magnitude = 0.9
        current_analysis.activation_capture.activation_sparsity = 0.8
        current_analysis.activation_capture.circuit_specialization = 0.7
        
        # Create current assessment
        current_assessment = Mock(spec=AssessmentResult)
        current_assessment.phq9_score = 8
        current_assessment.gad7_score = 6
        current_assessment.pss10_score = 7
        
        # Detect drift
        detection = await detector.detect_drift("test_persona", current_analysis, current_assessment, 10)
        
        assert detection is not None
        assert detection.persona_id == "test_persona"
        assert detection.current_day == 10
    
    def test_calculate_attention_drift(self, detector):
        """Test attention drift calculation."""
        attention_capture = Mock()
        attention_capture.self_reference_attention = 0.8
        attention_capture.emotional_salience = 0.6
        attention_capture.memory_integration = 0.7
        attention_capture.get_attention_summary.return_value = {"attention_entropy": 3.0}
        
        baseline_stats = {
            "mean": [0.5, 0.3, 0.4, 2.0],
            "std": [0.1, 0.1, 0.1, 0.5]
        }
        
        drift = detector._calculate_attention_drift(attention_capture, baseline_stats)
        assert isinstance(drift, dict)
        assert len(drift) > 0
    
    def test_determine_drift_direction(self, detector):
        """Test drift direction determination."""
        trait_drift = {"trait1": 0.2}
        clinical_drift = {"phq9": 0.3}
        mechanistic_drift = {"attention_0": 0.1}
        
        direction = detector._determine_drift_direction(trait_drift, clinical_drift, mechanistic_drift)
        assert direction in ["positive", "negative", "neutral"]


class TestCircuitTracker:
    """Test circuit tracker functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create circuit tracker instance."""
        return CircuitTracker()
    
    @pytest.fixture
    def mock_mechanistic_analysis(self):
        """Create mock mechanistic analysis."""
        analysis = Mock(spec=MechanisticAnalysis)
        analysis.analysis_timestamp = datetime.utcnow()
        
        analysis.attention_capture = Mock()
        analysis.attention_capture.self_reference_attention = 0.6
        analysis.attention_capture.emotional_salience = 0.4
        analysis.attention_capture.memory_integration = 0.7
        analysis.attention_capture.get_attention_summary.return_value = {"attention_entropy": 2.5}
        
        analysis.activation_capture = Mock()
        analysis.activation_capture.activation_magnitude = 0.8
        analysis.activation_capture.activation_sparsity = 0.6
        analysis.activation_capture.circuit_specialization = 0.5
        
        return analysis
    
    def test_neural_circuit_initialization(self):
        """Test neural circuit initialization."""
        circuit = NeuralCircuit("test_circuit", "Test Circuit", "Test description")
        assert circuit.circuit_id == "test_circuit"
        assert circuit.name == "Test Circuit"
        assert circuit.description == "Test description"
        assert len(circuit.activation_history) == 0
    
    def test_neural_circuit_add_activation(self):
        """Test adding activation to circuit."""
        circuit = NeuralCircuit("test_circuit", "Test Circuit", "Test description")
        timestamp = datetime.utcnow()
        
        circuit.add_activation(0.7, timestamp)
        assert len(circuit.activation_history) == 1
        assert circuit.activation_history[0] == 0.7
        assert circuit.total_activations == 1
    
    def test_neural_circuit_calculate_specialization(self):
        """Test circuit specialization calculation."""
        circuit = NeuralCircuit("test_circuit", "Test Circuit", "Test description")
        
        # Add some activations
        for i in range(5):
            circuit.add_activation(0.5 + i * 0.1, datetime.utcnow())
        
        specialization = circuit.calculate_specialization()
        assert isinstance(specialization, float)
        assert 0.0 <= specialization <= 1.0
    
    def test_neural_circuit_calculate_stability(self):
        """Test circuit stability calculation."""
        circuit = NeuralCircuit("test_circuit", "Test Circuit", "Test description")
        
        # Add some activations
        for i in range(5):
            circuit.add_activation(0.5, datetime.utcnow())  # Consistent activations
        
        stability = circuit.calculate_stability()
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0
    
    @pytest.mark.asyncio
    async def test_setup_circuit_tracking(self, tracker):
        """Test circuit tracking setup."""
        result = await tracker.setup_circuit_tracking("test_persona")
        assert result is True
        assert "test_persona" in tracker.persona_circuits
    
    @pytest.mark.asyncio
    async def test_track_circuit_activations(self, tracker, mock_mechanistic_analysis):
        """Test circuit activation tracking."""
        activations = await tracker.track_circuit_activations("test_persona", mock_mechanistic_analysis)
        assert isinstance(activations, dict)
        assert len(activations) > 0
    
    @pytest.mark.asyncio
    async def test_get_circuit_summary(self, tracker):
        """Test circuit summary retrieval."""
        await tracker.setup_circuit_tracking("test_persona")
        summary = await tracker.get_circuit_summary("test_persona")
        assert isinstance(summary, dict)
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_get_highly_active_circuits(self, tracker):
        """Test highly active circuits retrieval."""
        await tracker.setup_circuit_tracking("test_persona")
        active_circuits = await tracker.get_highly_active_circuits("test_persona")
        assert isinstance(active_circuits, list)
    
    @pytest.mark.asyncio
    async def test_identify_circuit_anomalies(self, tracker):
        """Test circuit anomaly identification."""
        await tracker.setup_circuit_tracking("test_persona")
        anomalies = await tracker.identify_circuit_anomalies("test_persona")
        assert isinstance(anomalies, dict)
        assert "highly_active" in anomalies
        assert "unstable" in anomalies


class TestInterventionEngine:
    """Test intervention engine functionality."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock()
        service.is_loaded = True
        service.model = Mock()
        service.tokenizer = Mock()
        service.generate_response = AsyncMock(return_value=("test response", {"tokens_generated": 10}))
        return service
    
    @pytest.fixture
    def engine(self, mock_llm_service):
        """Create intervention engine instance."""
        return InterventionEngine(mock_llm_service)
    
    def test_intervention_hook_initialization(self):
        """Test intervention hook initialization."""
        hook = InterventionHook(layer_idx=0, intervention_type="activation_patch")
        assert hook.layer_idx == 0
        assert hook.intervention_type == "activation_patch"
        assert len(hook.original_activations) == 0
        assert len(hook.intervened_activations) == 0
        assert not hook.intervention_applied
    
    @pytest.mark.asyncio
    async def test_setup_interventions_success(self, engine):
        """Test successful intervention setup."""
        result = await engine.setup_interventions()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_setup_interventions_failure(self, engine):
        """Test intervention setup failure."""
        engine.llm_service.is_loaded = False
        result = await engine.setup_interventions()
        assert result is False
    
    def test_remove_hooks(self, engine):
        """Test hook removal."""
        # Add some mock hooks
        mock_hook = Mock()
        mock_hook.remove = Mock()
        engine.hooks = [mock_hook]
        
        engine._remove_hooks()
        
        assert len(engine.hooks) == 0
        mock_hook.remove.assert_called_once()
    
    def test_calculate_response_similarity(self, engine):
        """Test response similarity calculation."""
        response1 = "I feel happy about this"
        response2 = "I feel happy about this situation"
        
        similarity = engine._calculate_response_similarity(response1, response2)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_calculate_semantic_similarity(self, engine):
        """Test semantic similarity calculation."""
        response1 = "I feel happy and excited"
        response2 = "I feel sad and worried"
        
        similarity = engine._calculate_semantic_similarity(response1, response2)
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_calculate_intervention_magnitude(self, engine):
        """Test intervention magnitude calculation."""
        # Add some mock hooks
        mock_hook = Mock()
        mock_hook.intervention_applied = True
        engine.hooks = [mock_hook, mock_hook]
        
        magnitude = engine._calculate_intervention_magnitude()
        assert isinstance(magnitude, float)
        assert 0.0 <= magnitude <= 1.0


class TestMechanisticVisualizer:
    """Test mechanistic visualizer functionality."""
    
    @pytest.fixture
    def visualizer(self):
        """Create mechanistic visualizer instance."""
        return MechanisticVisualizer()
    
    @pytest.fixture
    def mock_mechanistic_analysis(self):
        """Create mock mechanistic analysis."""
        analysis = Mock(spec=MechanisticAnalysis)
        analysis.analysis_timestamp = datetime.utcnow()
        
        analysis.attention_capture = Mock()
        analysis.attention_capture.attention_weights = [[0.5, 0.5], [0.3, 0.7]]
        
        analysis.activation_capture = Mock()
        analysis.activation_capture.activation_magnitude = 0.8
        
        return analysis
    
    @pytest.fixture
    def mock_drift_detection(self):
        """Create mock drift detection."""
        detection = Mock(spec=DriftDetection)
        detection.current_day = 10
        detection.drift_magnitude = 0.3
        detection.drift_direction = "positive"
        detection.drift_detected = True
        detection.significant_drift = False
        detection.affected_circuits = ["attention_0"]
        return detection
    
    @pytest.mark.asyncio
    async def test_add_analysis(self, visualizer, mock_mechanistic_analysis):
        """Test adding analysis to visualizer."""
        await visualizer.add_analysis("test_persona", mock_mechanistic_analysis)
        assert "test_persona" in visualizer.visualization_history
        assert len(visualizer.visualization_history["test_persona"]) == 1
    
    @pytest.mark.asyncio
    async def test_add_drift_detection(self, visualizer, mock_drift_detection):
        """Test adding drift detection to visualizer."""
        await visualizer.add_drift_detection("test_persona", mock_drift_detection)
        assert "test_persona" in visualizer.drift_timelines
        assert len(visualizer.drift_timelines["test_persona"]) == 1
    
    @pytest.mark.asyncio
    async def test_create_attention_heatmap(self, visualizer, mock_mechanistic_analysis):
        """Test attention heatmap creation."""
        await visualizer.add_analysis("test_persona", mock_mechanistic_analysis)
        heatmap = await visualizer.create_attention_heatmap("test_persona")
        assert heatmap is not None
    
    @pytest.mark.asyncio
    async def test_create_activation_timeline(self, visualizer, mock_mechanistic_analysis):
        """Test activation timeline creation."""
        await visualizer.add_analysis("test_persona", mock_mechanistic_analysis)
        timeline = await visualizer.create_activation_timeline("test_persona")
        assert timeline is not None
    
    @pytest.mark.asyncio
    async def test_create_drift_visualization(self, visualizer, mock_drift_detection):
        """Test drift visualization creation."""
        await visualizer.add_drift_detection("test_persona", mock_drift_detection)
        drift_viz = await visualizer.create_drift_visualization("test_persona")
        assert drift_viz is not None
    
    @pytest.mark.asyncio
    async def test_create_mechanistic_summary(self, visualizer, mock_mechanistic_analysis):
        """Test mechanistic summary creation."""
        await visualizer.add_analysis("test_persona", mock_mechanistic_analysis)
        summary = await visualizer.create_mechanistic_summary("test_persona")
        assert isinstance(summary, dict)
        assert "persona_id" in summary
        assert "total_analyses" in summary
    
    def test_calculate_entropy(self, visualizer):
        """Test entropy calculation."""
        matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        entropy = visualizer._calculate_entropy(matrix)
        assert isinstance(entropy, float)
        assert entropy >= 0.0
    
    def test_calculate_trend(self, visualizer):
        """Test trend calculation."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = visualizer._calculate_trend(values)
        assert isinstance(trend, float)
    
    def test_calculate_stability(self, visualizer):
        """Test stability calculation."""
        values = [0.5, 0.5, 0.5, 0.5, 0.5]  # Stable values
        stability = visualizer._calculate_stability(values)
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0


if __name__ == "__main__":
    pytest.main([__file__]) 