#!/usr/bin/env python3
"""
Simulation runner script for AI personality drift experiments.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.simulation_engine import SimulationEngine
from src.models.simulation import ExperimentalCondition
from src.core.config import config_manager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


async def run_single_condition(
    condition: ExperimentalCondition,
    config_name: str = "experimental_design"
) -> bool:
    """Run simulation for a single experimental condition."""
    try:
        logger.info(f"Starting simulation for condition: {condition.value}")
        
        # Initialize simulation engine
        engine = SimulationEngine()
        
        # Initialize simulation
        success = await engine.initialize_simulation(
            config_name=config_name,
            experimental_condition=condition
        )
        
        if not success:
            logger.error(f"Failed to initialize simulation for condition: {condition.value}")
            return False
        
        # Run simulation
        start_time = time.time()
        success = await engine.run_simulation()
        end_time = time.time()
        
        if success:
            # Get results
            results = await engine.get_simulation_results()
            
            logger.info(f"Simulation completed for condition: {condition.value}")
            logger.info(f"Duration: {end_time - start_time:.2f} seconds")
            logger.info(f"Events processed: {results.get('total_events_processed', 0)}")
            logger.info(f"Assessments completed: {results.get('total_assessments_completed', 0)}")
            
            # Save results
            await save_simulation_results(condition, results)
            
        else:
            logger.error(f"Simulation failed for condition: {condition.value}")
        
        # Cleanup
        await engine.cleanup()
        
        return success
        
    except Exception as e:
        logger.error(f"Error running simulation for condition {condition.value}: {e}")
        return False


async def run_all_conditions() -> None:
    """Run simulation for all experimental conditions."""
    conditions = [
        ExperimentalCondition.STRESS,
        ExperimentalCondition.NEUTRAL,
        ExperimentalCondition.MINIMAL
    ]
    
    results = {}
    
    for condition in conditions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running condition: {condition.value}")
        logger.info(f"{'='*50}")
        
        success = await run_single_condition(condition)
        results[condition.value] = success
        
        # Small delay between conditions
        await asyncio.sleep(2)
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SIMULATION SUMMARY")
    logger.info(f"{'='*50}")
    
    for condition, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        logger.info(f"{condition}: {status}")
    
    successful_conditions = sum(1 for success in results.values() if success)
    total_conditions = len(conditions)
    
    logger.info(f"\nOverall: {successful_conditions}/{total_conditions} conditions completed successfully")


async def save_simulation_results(condition: ExperimentalCondition, results: dict) -> None:
    """Save simulation results to file."""
    try:
        from datetime import datetime
        import json
        
        # Create results directory
        results_dir = Path("data/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"simulation_{condition.value}_{timestamp}.json"
        filepath = results_dir / filename
        
        # Save results
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


async def main():
    """Main function."""
    try:
        logger.info("AI Personality Drift Simulation")
        logger.info("=" * 50)
        
        # Check if we should run all conditions or just one
        if len(sys.argv) > 1:
            condition_name = sys.argv[1].upper()
            try:
                condition = ExperimentalCondition(condition_name)
                await run_single_condition(condition)
            except ValueError:
                logger.error(f"Invalid condition: {condition_name}")
                logger.info("Valid conditions: STRESS, NEUTRAL, MINIMAL")
                return
        else:
            # Run all conditions
            await run_all_conditions()
        
        logger.info("Simulation runner completed")
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 