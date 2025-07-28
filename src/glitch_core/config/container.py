"""
Dependency injection container for Glitch Core.
"""

from dependency_injector import containers, providers
from typing import Optional

from .settings import Settings


class Container(containers.DeclarativeContainer):
    """Dependency injection container for Glitch Core components."""
    
    # Configuration
    config = providers.Singleton(Settings)
    
    # Core components - using factory functions to avoid circular imports
    memory_manager = providers.Singleton(
        providers.Factory(lambda settings: _create_memory_manager(settings),
                         settings=config)
    )
    
    reflection_engine = providers.Singleton(
        providers.Factory(lambda settings: _create_reflection_engine(settings),
                         settings=config)
    )
    
    # Main engine
    drift_engine = providers.Singleton(
        providers.Factory(lambda settings, memory_manager, reflection_engine: 
                         _create_drift_engine(settings, memory_manager, reflection_engine),
                         settings=config,
                         memory_manager=memory_manager,
                         reflection_engine=reflection_engine)
    )


def _create_memory_manager(settings: Settings):
    """Factory function to create MemoryManager."""
    from ..core.memory import MemoryManager
    return MemoryManager(settings)


def _create_reflection_engine(settings: Settings):
    """Factory function to create ReflectionEngine."""
    from ..core.llm import ReflectionEngine
    return ReflectionEngine(settings)


def _create_drift_engine(settings: Settings, memory_manager, reflection_engine):
    """Factory function to create DriftEngine."""
    from ..core.drift_engine.engine import DriftEngine
    return DriftEngine(settings, memory_manager, reflection_engine)


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def get_settings() -> Settings:
    """Get settings from the container."""
    return get_container().config()


def get_drift_engine():
    """Get drift engine from the container."""
    return get_container().drift_engine()


def get_memory_manager():
    """Get memory manager from the container."""
    return get_container().memory_manager()


def get_reflection_engine():
    """Get reflection engine from the container."""
    return get_container().reflection_engine() 