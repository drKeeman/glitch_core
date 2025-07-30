#!/usr/bin/env python3
"""
Test simulation setup script for Makefile.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.services.simulation_engine import SimulationEngine
from src.models.simulation import ExperimentalCondition


async def main():
    """Test simulation setup."""
    try:
        engine = SimulationEngine()
        success = await engine.initialize_simulation('experimental_design', ExperimentalCondition.CONTROL)
        print(f'✅ Simulation setup: {success}')
        if success:
            await engine.cleanup()
        return 0 if success else 1
    except Exception as e:
        print(f'❌ Simulation test error: {e}')
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 