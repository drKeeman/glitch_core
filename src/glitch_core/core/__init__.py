"""Core module for Glitch Core with Phase 2 enhancements."""

from .types import *
from .exceptions import *

# Phase 2.1: Memory Management Optimizations
from .memory import (
    MemoryManager, MemoryRecord,
    MemoryCompressor, CompressedMemory,
    TemporalDecay, DecayConfig,
    RelevanceScorer, RelevanceConfig, RelevanceScore,
    MemoryVisualizer, MemoryVisualizationData
)

# Phase 2.2: Analysis Capabilities Enhancement
from .analysis import (
    StabilityAnalyzer, PatternDetector, TemporalAnalyzer,
    DriftAnalyzer, DriftPattern, StabilityAnalysis, PersonalityEvolution
)

# Phase 2.3: Intervention Framework Enhancement
from .interventions import (
    InterventionFramework, Intervention, InterventionImpact,
    InterventionTemplate, InterventionType, InterventionIntensity
)

__all__ = [
    # Types
    "EmotionalState", "EventData", "PersonaConfig", "DriftProfile",
    "MemoryRecord", "InterventionData", "AnalysisResult", "SimulationState",
    "APIResponse", "WebSocketMessage", "ExperimentId", "MemoryId", "UserId",
    "Timestamp", "JSONValue", "ConfigValue", "LogLevel", "HTTPStatusCode",
    "URL", "ModelName", "CollectionName", "DatabaseURL", "APIEndpoint",
    "WebSocketEndpoint", "ErrorMessage", "SuccessMessage", "WarningMessage",
    "InfoMessage", "DebugMessage", "CorrelationId", "RequestId", "SessionId",
    "TokenCount", "ConfidenceScore", "EmotionalWeight", "DecayRate",
    "EpochNumber", "EventCount", "MemoryLimit", "Timeout", "RetryCount",
    "CircuitBreakerState", "HealthStatus", "Version", "Environment",
    "FeatureFlag", "MetricValue", "MetricName", "MetricLabels",
    "TraceId", "SpanId", "ParentSpanId", "SpanName", "SpanKind",
    "SpanAttributes", "SpanEvents", "SpanLinks", "SpanStatus",
    "SpanStatusCode", "SpanStatusMessage", "SpanStartTime", "SpanEndTime",
    "SpanDuration", "LogRecord", "LogMessage", "LogContext", "LogTimestamp",
    "LogSource", "LogModule", "LogFunction", "LogLine", "LogException",
    "LogStackTrace", "LogCorrelationId", "LogRequestId", "LogUserId",
    "LogSessionId", "LogTraceId", "LogSpanId", "LogParentSpanId",
    "LogSpanName", "LogSpanKind", "LogSpanAttributes", "LogSpanEvents",
    "LogSpanLinks", "LogSpanStatus", "LogSpanStatusCode", "LogSpanStatusMessage",
    "LogSpanStartTime", "LogSpanEndTime", "LogSpanDuration",
    
    # Exceptions
    "GlitchCoreError", "ConfigurationError", "SimulationError", "LLMConnectionError",
    "MemoryStorageError", "DatabaseConnectionError", "ValidationError", "InterventionError",
    "AnalysisError", "WebSocketError", "AuthenticationError", "AuthorizationError",
    "RateLimitError", "ResourceNotFoundError", "ServiceUnavailableError", "TimeoutError",
    "DataIntegrityError", "ExternalServiceError", "CircuitBreakerError", "RetryableError",
    "NonRetryableError",
    
    # Phase 2.1: Memory Management Optimizations
    "MemoryManager", "MemoryRecord",
    "MemoryCompressor", "CompressedMemory",
    "TemporalDecay", "DecayConfig",
    "RelevanceScorer", "RelevanceConfig", "RelevanceScore",
    "MemoryVisualizer", "MemoryVisualizationData",
    
    # Phase 2.2: Analysis Capabilities Enhancement
    "StabilityAnalyzer", "PatternDetector", "TemporalAnalyzer",
    "DriftAnalyzer", "DriftPattern", "StabilityAnalysis", "PersonalityEvolution",
    
    # Phase 2.3: Intervention Framework Enhancement
    "InterventionFramework", "Intervention", "InterventionImpact",
    "InterventionTemplate", "InterventionType", "InterventionIntensity"
] 