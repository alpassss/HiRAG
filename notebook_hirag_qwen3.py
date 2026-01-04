"""
HiRAG with Qwen3 8B and Qwen3 Embedding for Kaggle Notebook
============================================================
This script is designed to run HiRAG system in Kaggle Notebook environment with:
- Qwen3-8B-Base model from Hugging Face (non-instruct version)
- Qwen3-Embedding-0.6B for text embeddings
- Progress bars and timing for all major steps
- Notebook-compatible async execution
- GPU support (optimized for H100 80GB)

Usage in Notebook:
    %run notebook_hirag_qwen3.py
"""

import os
import sys
import json
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
import nest_asyncio

# Import HiRAG components
from hirag import HiRAG, QueryParam
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from hirag._storage import NetworkXStorage

# Enable nested asyncio for Jupyter/Kaggle notebook environment
nest_asyncio.apply()

# ============================================================================
# CONFIGURATION SECTION - Modify parameters here
# ============================================================================

# File paths
DATA_FILE = "./eval/datasets/cs/cs_unique_contexts.json"
WORKING_DIR = "./hirag_qwen3_workdir"

# Model configuration
LLM_MODEL_NAME = "Qwen/Qwen3-8B-Base"  # Non-instruct version
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Model parameters
EMBEDDING_DIM = 1024  # Qwen3-Embedding-0.6B default dimension
MAX_TOKEN_SIZE = 8192
EMBEDDING_BATCH_SIZE = 16  # Batch size for embedding computation
MAX_NEW_TOKENS = 2048  # Maximum tokens to generate

# HiRAG configuration
ENABLE_LLM_CACHE = True
ENABLE_HIERARCHICAL_MODE = True
ENABLE_NAIVE_RAG = True
EMBEDDING_BATCH_NUM = 6
EMBEDDING_FUNC_MAX_ASYNC = 8
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100

# Data loading configuration
NUM_CONTEXTS_TO_LOAD = None  # None = load all, or set to a number like 100 for testing

# GPU configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16  # Use bfloat16 for H100/A100, float16 for older GPUs

# Query configuration for testing
TEST_QUERY = "What are the main concepts discussed in computer science?"
QUERY_MODE = "hi"  # Options: "hi", "naive", "hi_nobridge", "hi_local", "hi_global", "hi_bridge"

# ============================================================================
# Setup logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Load models with progress tracking
# ============================================================================

print("="*80)
print("HiRAG System Initialization with Qwen3 Models")
print("="*80)
print(f"Device: {DEVICE}")
print(f"LLM Model: {LLM_MODEL_NAME}")
print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"Data File: {DATA_FILE}")
print(f"Working Directory: {WORKING_DIR}")
print("="*80)

# Timer class for tracking execution time
class Timer:
    def __init__(self, description):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️  Starting: {self.description}")
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"✅ Completed: {self.description} (Time: {elapsed:.2f}s)")

# ============================================================================
# Load Hugging Face Models
# ============================================================================

with Timer("Loading models from Hugging Face"):
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
    
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    embedding_model.eval()
    
    # Load LLM model
    print(f"Loading LLM model: {LLM_MODEL_NAME}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    llm_model.eval()
    
    # Create text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
    )

print(f"✅ Models loaded successfully on {DEVICE}")

# ============================================================================
# Define embedding function for HiRAG
# ============================================================================

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""
    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func
    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=MAX_TOKEN_SIZE)
async def qwen3_embedding(texts: List[str]) -> np.ndarray:
    """Generate embeddings using Qwen3-Embedding model"""
    
    # Process in batches to manage memory
    all_embeddings = []
    
    try:
        with torch.no_grad():
            for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
                batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
                
                # Tokenize
                inputs = embedding_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_TOKEN_SIZE,
                    return_tensors="pt"
                ).to(DEVICE)
                
                # Generate embeddings
                outputs = embedding_model(**inputs)
                
                # Use mean pooling on the last hidden state
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Move to CPU and convert to numpy
                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
                
                # Clear GPU cache periodically
                if i % (EMBEDDING_BATCH_SIZE * 10) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
    except Exception as e:
        logger.error(f"Error in embedding generation: {e}")
        raise

# ============================================================================
# Define LLM function for HiRAG
# ============================================================================

async def qwen3_llm_if_cache(
    prompt, 
    system_prompt=None, 
    history_messages=[], 
    **kwargs
) -> str:
    """Generate text using Qwen3-8B-Base model with caching"""
    
    try:
        # Build the full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        # Add history messages
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            full_prompt += f"{role}: {content}\n"
        
        # Add current prompt
        full_prompt += f"user: {prompt}\nassistant:"
        
        # Check cache
        hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        
        if hashing_kv is not None:
            args_hash = compute_args_hash(LLM_MODEL_NAME, messages)
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]
        
        # Generate response
        with torch.no_grad():
            outputs = text_generator(
                full_prompt,
                max_new_tokens=kwargs.get("max_tokens", MAX_NEW_TOKENS),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.95),
                pad_token_id=llm_tokenizer.eos_token_id,
            )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        # Remove the prompt from the output
        if generated_text.startswith(full_prompt):
            response = generated_text[len(full_prompt):].strip()
        else:
            response = generated_text.strip()
        
        # Cache the response
        if hashing_kv is not None:
            await hashing_kv.upsert(
                {args_hash: {"return": response, "model": LLM_MODEL_NAME}}
            )
        
        # Clear GPU cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response
    except Exception as e:
        logger.error(f"Error in LLM generation: {e}")
        # Return a fallback response
        return f"Error generating response: {str(e)}"

# ============================================================================
# Initialize HiRAG system with progress tracking
# ============================================================================

with Timer("Initializing HiRAG system"):
    graph_func = HiRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=ENABLE_LLM_CACHE,
        embedding_func=qwen3_embedding,
        best_model_func=qwen3_llm_if_cache,
        cheap_model_func=qwen3_llm_if_cache,
        enable_hierachical_mode=ENABLE_HIERARCHICAL_MODE,  # Note: 'hierachical' spelling is intentional in HiRAG API
        embedding_batch_num=EMBEDDING_BATCH_NUM,
        embedding_func_max_async=EMBEDDING_FUNC_MAX_ASYNC,
        enable_naive_rag=ENABLE_NAIVE_RAG,
        graph_storage_cls=NetworkXStorage,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
    )

# ============================================================================
# Load and process data with progress tracking
# ============================================================================

with Timer("Loading data from JSON file"):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        contexts = json.load(f)
    
    # Limit contexts if specified
    if NUM_CONTEXTS_TO_LOAD is not None:
        contexts = contexts[:NUM_CONTEXTS_TO_LOAD]
        print(f"Using first {NUM_CONTEXTS_TO_LOAD} contexts for testing")
    else:
        print(f"Loaded {len(contexts)} contexts")

# ============================================================================
# Insert contexts into knowledge graph with progress tracking
# ============================================================================

print("\n" + "="*80)
print("Building Knowledge Graph")
print("="*80)

with Timer("Inserting contexts into knowledge graph"):
    # Add progress tracking to insert operation
    print(f"Processing {len(contexts)} contexts...")
    
    # Create a custom progress bar wrapper
    original_insert = graph_func.insert
    
    def insert_with_progress(data):
        with tqdm(total=100, desc="Knowledge Graph Construction") as pbar:
            # We can't directly track internal progress, but we show activity
            result = original_insert(data)
            pbar.update(100)
            return result
    
    insert_with_progress(contexts)

print("\n✅ Knowledge graph built successfully!")

# ============================================================================
# Perform test query with progress tracking
# ============================================================================

print("\n" + "="*80)
print("Testing Query System")
print("="*80)

with Timer(f"Performing query in '{QUERY_MODE}' mode"):
    print(f"Query: {TEST_QUERY}")
    print(f"Mode: {QUERY_MODE}")
    
    response = graph_func.query(
        TEST_QUERY, 
        param=QueryParam(mode=QUERY_MODE)
    )

print("\n" + "="*80)
print("Query Response")
print("="*80)
print(response)

# ============================================================================
# System Summary
# ============================================================================

print("\n" + "="*80)
print("System Summary")
print("="*80)
print(f"✅ LLM Model: {LLM_MODEL_NAME}")
print(f"✅ Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"✅ Contexts processed: {len(contexts)}")
print(f"✅ Working directory: {WORKING_DIR}")
print(f"✅ Device: {DEVICE}")
print(f"✅ GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
print(f"✅ GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
print("="*80)

print("\n🎉 HiRAG system is ready for queries!")
print("\nTo perform additional queries, use:")
print("  response = graph_func.query('Your question here', param=QueryParam(mode='hi'))")
print("\nAvailable query modes: 'hi', 'naive', 'hi_nobridge', 'hi_local', 'hi_global', 'hi_bridge'")
