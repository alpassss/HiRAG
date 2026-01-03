#!/usr/bin/env python3
"""
Test script to verify that local models work without OpenAI API
"""

import asyncio
import numpy as np
from hirag.local_embedding import local_embedding
from hirag.local_llm import local_gpt_4o_mini_complete

async def test_local_embedding():
    print("Testing local embedding function...")
    try:
        texts = ["Hello world", "This is a test", "Another sentence"]
        embeddings = await local_embedding(texts)
        print(f"✓ Successfully generated embeddings for {len(texts)} texts")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Embedding dtype: {embeddings.dtype}")
        print(f"  First embedding (first 10 elements): {embeddings[0][:10]}")
        return True
    except Exception as e:
        print(f"✗ Error in local embedding: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_local_llm():
    print("\nTesting local LLM function...")
    try:
        prompt = "What is the capital of France?"
        response = await local_gpt_4o_mini_complete(prompt)
        print(f"✓ Successfully generated response: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ Error in local LLM: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("Testing local models without OpenAI API...")
    
    embedding_success = await test_local_embedding()
    llm_success = await test_local_llm()
    
    if embedding_success and llm_success:
        print("\n✓ All tests passed! Local models are working correctly.")
    else:
        print("\n✗ Some tests failed.")

if __name__ == "__main__":
    asyncio.run(main())