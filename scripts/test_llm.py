#!/usr/bin/env python3
"""
Test LLM connection script for Makefile.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.setup_ollama import test_ollama_connection


async def main():
    """Test LLM connection."""
    success = await test_ollama_connection()
    if success:
        print("✅ LLM test passed")
        return 0
    else:
        print("❌ LLM test failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 