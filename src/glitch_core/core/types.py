"""
Type definitions and aliases for Glitch Core.
"""

from typing import TypeAlias, Dict, List, Any, Union, Optional
from datetime import datetime


# Emotional state type
EmotionalState: TypeAlias = Dict[str, float]

# Event data type
EventData: TypeAlias = Dict[str, Any]

# Persona configuration type
PersonaConfig: TypeAlias = Dict[str, Any]

# Drift profile type
DriftProfile: TypeAlias = Dict[str, Any]

# Memory record type
MemoryRecord: TypeAlias = Dict[str, Any]

# Intervention data type
InterventionData: TypeAlias = Dict[str, Any]

# Analysis result type
AnalysisResult: TypeAlias = Dict[str, Any]

# Simulation state type
SimulationState: TypeAlias = Dict[str, Any]

# API response type
APIResponse: TypeAlias = Dict[str, Any]

# WebSocket message type
WebSocketMessage: TypeAlias = Dict[str, Any]

# Experiment ID type
ExperimentId: TypeAlias = str

# Memory ID type
MemoryId: TypeAlias = str

# User ID type
UserId: TypeAlias = str

# Timestamp type
Timestamp: TypeAlias = Union[datetime, float, str]

# JSON serializable type
JSONValue: TypeAlias = Union[str, int, float, bool, None, List[Any], Dict[str, Any]]

# Configuration type
ConfigValue: TypeAlias = Union[str, int, float, bool, List[str], Dict[str, Any]]

# Log level type
LogLevel: TypeAlias = str

# HTTP status code type
HTTPStatusCode: TypeAlias = int

# URL type
URL: TypeAlias = str

# Model name type
ModelName: TypeAlias = str

# Collection name type
CollectionName: TypeAlias = str

# Database URL type
DatabaseURL: TypeAlias = str

# API endpoint type
APIEndpoint: TypeAlias = str

# WebSocket endpoint type
WebSocketEndpoint: TypeAlias = str

# Error message type
ErrorMessage: TypeAlias = str

# Success message type
SuccessMessage: TypeAlias = str

# Warning message type
WarningMessage: TypeAlias = str

# Info message type
InfoMessage: TypeAlias = str

# Debug message type
DebugMessage: TypeAlias = str

# Correlation ID type
CorrelationId: TypeAlias = str

# Request ID type
RequestId: TypeAlias = str

# Session ID type
SessionId: TypeAlias = str

# Token count type
TokenCount: TypeAlias = int

# Confidence score type
ConfidenceScore: TypeAlias = float

# Emotional weight type
EmotionalWeight: TypeAlias = float

# Decay rate type
DecayRate: TypeAlias = float

# Epoch number type
EpochNumber: TypeAlias = int

# Event count type
EventCount: TypeAlias = int

# Memory limit type
MemoryLimit: TypeAlias = int

# Timeout type
Timeout: TypeAlias = float

# Retry count type
RetryCount: TypeAlias = int

# Circuit breaker state type
CircuitBreakerState: TypeAlias = str

# Health status type
HealthStatus: TypeAlias = str

# Version type
Version: TypeAlias = str

# Environment type
Environment: TypeAlias = str

# Feature flag type
FeatureFlag: TypeAlias = bool

# Metric value type
MetricValue: TypeAlias = Union[int, float]

# Metric name type
MetricName: TypeAlias = str

# Metric labels type
MetricLabels: TypeAlias = Dict[str, str]

# Trace ID type
TraceId: TypeAlias = str

# Span ID type
SpanId: TypeAlias = str

# Parent span ID type
ParentSpanId: TypeAlias = str

# Span name type
SpanName: TypeAlias = str

# Span kind type
SpanKind: TypeAlias = str

# Span attributes type
SpanAttributes: TypeAlias = Dict[str, Any]

# Span events type
SpanEvents: TypeAlias = List[Dict[str, Any]]

# Span links type
SpanLinks: TypeAlias = List[Dict[str, Any]]

# Span status type
SpanStatus: TypeAlias = str

# Span status code type
SpanStatusCode: TypeAlias = str

# Span status message type
SpanStatusMessage: TypeAlias = str

# Span start time type
SpanStartTime: TypeAlias = float

# Span end time type
SpanEndTime: TypeAlias = float

# Span duration type
SpanDuration: TypeAlias = float

# Log record type
LogRecord: TypeAlias = Dict[str, Any]

# Log level type
LogLevel: TypeAlias = str

# Log message type
LogMessage: TypeAlias = str

# Log context type
LogContext: TypeAlias = Dict[str, Any]

# Log timestamp type
LogTimestamp: TypeAlias = float

# Log source type
LogSource: TypeAlias = str

# Log module type
LogModule: TypeAlias = str

# Log function type
LogFunction: TypeAlias = str

# Log line type
LogLine: TypeAlias = int

# Log exception type
LogException: TypeAlias = str

# Log stack trace type
LogStackTrace: TypeAlias = str

# Log correlation ID type
LogCorrelationId: TypeAlias = str

# Log request ID type
LogRequestId: TypeAlias = str

# Log user ID type
LogUserId: TypeAlias = str

# Log session ID type
LogSessionId: TypeAlias = str

# Log trace ID type
LogTraceId: TypeAlias = str

# Log span ID type
LogSpanId: TypeAlias = str

# Log parent span ID type
LogParentSpanId: TypeAlias = str

# Log span name type
LogSpanName: TypeAlias = str

# Log span kind type
LogSpanKind: TypeAlias = str

# Log span attributes type
LogSpanAttributes: TypeAlias = Dict[str, Any]

# Log span events type
LogSpanEvents: TypeAlias = List[Dict[str, Any]]

# Log span links type
LogSpanLinks: TypeAlias = List[Dict[str, Any]]

# Log span status type
LogSpanStatus: TypeAlias = str

# Log span status code type
LogSpanStatusCode: TypeAlias = str

# Log span status message type
LogSpanStatusMessage: TypeAlias = str

# Log span start time type
LogSpanStartTime: TypeAlias = float

# Log span end time type
LogSpanEndTime: TypeAlias = float

# Log span duration type
LogSpanDuration: TypeAlias = float 