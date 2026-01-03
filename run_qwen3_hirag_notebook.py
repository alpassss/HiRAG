#!/usr/bin/env python3
"""
Qwen3-8B HiRAG System for Kaggle Notebook

This script implements a complete HiRAG system using Qwen3-8B model and Hugging Face 
embedding models optimized for H100 GPU in notebook environments.

Features:
- Uses Qwen/Qwen3-8B (base model, not instruct)
- Uses all-mpnet-base-v2 embedding model optimized for H100
- Progress bars for all operations
- Timing for each step
- Asynchronous operations with notebook compatibility
- Processes knowledge graph using eval/datasets/cs/cs_unique_contexts.json
- Optimized for H100 GPU (80GB)
"""

import os
import sys
import asyncio
import nest_asyncio
import torch
import gc

# Apply nest_asyncio to allow nested event loops in Jupyter notebooks
nest_asyncio.apply()

def main():
    print("Starting Qwen3-8B HiRAG System for Kaggle Notebook...")
    
    # Verify GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("GPU not available")
        
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Import the Qwen3 HiRAG system
    try:
        from qwen3_notebook_system import run_notebook_pipeline
        print("Qwen3 HiRAG system imported successfully!")
    except ImportError as e:
        print(f"Failed to import qwen3_notebook_system: {e}")
        print("Make sure the qwen3_notebook_system.py file is in the current directory")
        return
    
    # Define your configuration parameters
    config_params = {
        "llm_model_name": "Qwen/Qwen3-8B",  # Base Qwen3 8B model (not instruct)
        "embedding_model_name": "sentence-transformers/all-mpnet-base-v2",  # Better embedding model for H100
        "data_file": "/workspace/eval/datasets/cs/cs_unique_contexts.json",  # Knowledge graph data
        "num_contexts": 10,  # Number of contexts to process (adjust based on GPU memory)
        "queries": [
            "What are the key concepts in machine learning with Spark?",
            "Explain the architecture of a machine learning system",
            "How does collaborative filtering work in recommendation engines?"
        ]
    }

    print("\nConfiguration parameters:")
    for key, value in config_params.items():
        if key == "queries":
            print(f"  {key}: {len(value)} queries")
        else:
            print(f"  {key}: {value}")
    
    # Run the complete HiRAG pipeline
    print("\nStarting Qwen3 HiRAG pipeline...")
    print("This may take several minutes depending on model loading and data processing")

    try:
        results = run_notebook_pipeline(
            llm_model_name=config_params["llm_model_name"],
            embedding_model_name=config_params["embedding_model_name"],
            data_file=config_params["data_file"],
            num_contexts=config_params["num_contexts"],
            queries=config_params["queries"]
        )
        
        print("\nPipeline completed successfully!")
        print(f"Received {len(results)} responses")
        
        # Display results
        for i, (query, result) in enumerate(zip(config_params["queries"], results)):
            print(f"\n--- Query {i+1} ---")
            print(f"Question: {query}")
            print(f"Answer: {result}")
            print("-" * 50)
        
    except Exception as e:
        print(f"Error occurred during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Clean up GPU memory if needed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\nGPU memory cleared")

def run_custom_pipeline(
    llm_model_name: str = "Qwen/Qwen3-8B",
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    data_file: str = "/workspace/eval/datasets/cs/cs_unique_contexts.json",
    num_contexts: int = 5,
    queries: list = None
):
    """
    Function to run a custom pipeline with specified parameters
    """
    if queries is None:
        queries = [
            "Explain neural network architectures",
            "What is the difference between supervised and unsupervised learning?",
            "How do decision trees work in machine learning?"
        ]
    
    print(f"Running custom pipeline with {num_contexts} contexts and {len(queries)} queries...")
    
    try:
        results = run_notebook_pipeline(
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            data_file=data_file,
            num_contexts=num_contexts,
            queries=queries
        )
        
        print(f"Custom pipeline completed with {len(results)} results")
        return results
        
    except Exception as e:
        print(f"Error in custom pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()