#!/usr/bin/env python3
"""
Setup script for Ollama service and model initialization.
"""

import asyncio
import httpx
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_ollama():
    """Setup Ollama service and pull required model."""
    try:
        logger.info("Setting up Ollama service...")
        
        # Create HTTP client
        async with httpx.AsyncClient(timeout=60.0) as client:
            base_url = settings.OLLAMA_URL
            model_name = "llama3.1:8b"
            
            # Check if Ollama is running
            try:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    logger.info("✅ Ollama service is running")
                else:
                    logger.error(f"❌ Ollama service not responding: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"❌ Cannot connect to Ollama service: {e}")
                logger.info("Make sure Ollama is running: docker-compose up ollama")
                return False
            
            # Check available models
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            logger.info(f"Available models: {available_models}")
            
            if model_name in available_models:
                logger.info(f"✅ Model {model_name} is already available")
                return True
            
            # Pull the model
            logger.info(f"📥 Pulling model {model_name}...")
            pull_response = await client.post(
                f"{base_url}/api/pull",
                json={"name": model_name}
            )
            
            if pull_response.status_code == 200:
                logger.info(f"✅ Model {model_name} pulled successfully")
                return True
            else:
                logger.error(f"❌ Failed to pull model: {pull_response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error setting up Ollama: {e}")
        return False


async def test_ollama_connection():
    """Test Ollama connection and model availability."""
    try:
        logger.info("Testing Ollama connection...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            base_url = settings.OLLAMA_URL
            model_name = "llama3.1:8b"
            
            # Test simple generation
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello, how are you?",
                    "stream": False,
                    "options": {
                        "num_predict": 10,
                        "temperature": 0.7
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                logger.info(f"✅ Ollama test successful: {generated_text}")
                return True
            else:
                logger.error(f"❌ Ollama test failed: {response.text}")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error testing Ollama: {e}")
        return False


async def main():
    """Main setup function."""
    logger.info("🚀 Setting up Ollama for Glitch Core")
    logger.info("=" * 50)
    
    # Setup Ollama
    if await setup_ollama():
        logger.info("✅ Ollama setup completed successfully")
        
        # Test connection
        if await test_ollama_connection():
            logger.info("✅ Ollama is ready for use!")
            logger.info("You can now run your simulation with live LLM support.")
        else:
            logger.error("❌ Ollama test failed")
    else:
        logger.error("❌ Ollama setup failed")
        logger.info("Please check that:")
        logger.info("1. Docker is running")
        logger.info("2. Ollama service is started: docker-compose up ollama")
        logger.info("3. You have sufficient disk space for the model")


if __name__ == "__main__":
    asyncio.run(main()) 