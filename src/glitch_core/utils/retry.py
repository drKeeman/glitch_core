"""
Retry utilities with circuit breaker pattern for external service calls.
"""

import asyncio
import time
from typing import Callable, Any, Optional, TypeVar, Awaitable
from functools import wraps
from enum import Enum

from ..core.exceptions import (
    RetryableError, NonRetryableError, CircuitBreakerError,
    TimeoutError, ExternalServiceError
)


T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker implementation for external service calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError(
                    "Circuit breaker is open",
                    context={
                        "state": self.state.value,
                        "failure_count": self.failure_count,
                        "last_failure_time": self.last_failure_time
                    }
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute async function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerError(
                    "Circuit breaker is open",
                    context={
                        "state": self.state.value,
                        "failure_count": self.failure_count,
                        "last_failure_time": self.last_failure_time
                    }
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    retryable_exceptions: tuple = (RetryableError, TimeoutError, ExternalServiceError)
):
    """
    Retry decorator for functions that may fail transiently.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        retryable_exceptions: Tuple of exceptions that should trigger retries
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Calculate delay
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        raise
                    
                    # Calculate delay
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay
                    
                    await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        # Return appropriate wrapper based on if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Timeout decorator for functions that may hang.
    
    Args:
        timeout_seconds: Maximum time to wait for function completion
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return asyncio.run(asyncio.wait_for(
                    asyncio.create_task(func(*args, **kwargs)),
                    timeout=timeout_seconds
                ))
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function timed out after {timeout_seconds} seconds",
                    context={"function": func.__name__}
                )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"Function timed out after {timeout_seconds} seconds",
                    context={"function": func.__name__}
                )
        
        # Return appropriate wrapper based on if function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator 