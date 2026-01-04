"""
Example Usage in Kaggle Notebook Cell
======================================
Copy and paste this into a Kaggle notebook cell to run the HiRAG system.
"""

# ============================================================================
# Cell 1: Install Dependencies
# ============================================================================

# !pip install -e . -q
# Or if you need specific packages:
# !pip install torch transformers accelerate sentencepiece nest_asyncio tqdm -q

# ============================================================================
# Cell 2: Run the main script with default configuration
# ============================================================================

# %run notebook_hirag_qwen3.py

# This will:
# 1. Load Qwen3-8B-Base and Qwen3-Embedding-0.6B models
# 2. Load contexts from eval/datasets/cs/cs_unique_contexts.json
# 3. Build the knowledge graph
# 4. Perform a test query

# ============================================================================
# Cell 3: Perform custom queries (after running Cell 2)
# ============================================================================

# Example 1: Ask a question with hierarchical mode
response = graph_func.query(
    "What are the key algorithms discussed in computer science?",
    param=QueryParam(mode="hi")
)
print("Hierarchical Mode Response:")
print(response)
print("\n" + "="*80 + "\n")

# Example 2: Try naive RAG mode
response_naive = graph_func.query(
    "Explain the main programming concepts",
    param=QueryParam(mode="naive")
)
print("Naive RAG Response:")
print(response_naive)
print("\n" + "="*80 + "\n")

# Example 3: Local knowledge only
response_local = graph_func.query(
    "What entities are related to programming languages?",
    param=QueryParam(mode="hi_local")
)
print("Local Knowledge Response:")
print(response_local)

# ============================================================================
# Cell 4: Modify configuration and reload (optional)
# ============================================================================

# If you want to test with fewer contexts or different settings:

# Step 1: Modify the configuration at the top of notebook_hirag_qwen3.py
# For example, change:
#   NUM_CONTEXTS_TO_LOAD = 50  # Test with only 50 contexts
#   TEST_QUERY = "Your custom query here"

# Step 2: Re-run the script
# %run notebook_hirag_qwen3.py

# ============================================================================
# Cell 5: Check GPU memory usage
# ============================================================================

import torch

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved GPU Memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"Free GPU Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f} GB")
else:
    print("No GPU available")

# ============================================================================
# Cell 6: Advanced - Batch queries
# ============================================================================

# Process multiple queries
queries = [
    "What are the fundamental concepts in computer science?",
    "Explain data structures mentioned in the text",
    "What programming paradigms are discussed?",
]

print("Processing batch queries...")
for i, query in enumerate(queries, 1):
    print(f"\n{'='*80}")
    print(f"Query {i}: {query}")
    print('='*80)
    response = graph_func.query(query, param=QueryParam(mode="hi"))
    print(response)

# ============================================================================
# Cell 7: Compare different query modes
# ============================================================================

test_query = "What are the main topics covered in the documents?"

modes = ["hi", "naive", "hi_local", "hi_global"]

print("Comparing different query modes:\n")
for mode in modes:
    print(f"\n{'='*80}")
    print(f"Mode: {mode}")
    print('='*80)
    response = graph_func.query(test_query, param=QueryParam(mode=mode))
    print(response[:500] + "..." if len(response) > 500 else response)

# ============================================================================
# Cell 8: Clean up (optional)
# ============================================================================

# To free up GPU memory when done:
import gc

# Delete models
del llm_model, embedding_model, text_generator
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("GPU memory cleared")

# ============================================================================
# Notes:
# ============================================================================

# 1. First time running will download models (~18GB total), which takes time
# 2. Knowledge graph construction can take 10-30 minutes depending on data size
# 3. Subsequent queries are faster due to caching
# 4. Adjust NUM_CONTEXTS_TO_LOAD in the config for faster testing
# 5. Monitor GPU memory to avoid OOM errors

# ============================================================================
# Troubleshooting:
# ============================================================================

# If you get OOM (Out of Memory) errors:
# 1. Reduce NUM_CONTEXTS_TO_LOAD to 50 or 100
# 2. Reduce EMBEDDING_BATCH_SIZE to 8
# 3. Reduce CHUNK_TOKEN_SIZE to 800
# 4. Use smaller models (edit LLM_MODEL_NAME and EMBEDDING_MODEL_NAME)

# If models fail to load:
# 1. Check internet connection
# 2. Verify Hugging Face hub is accessible
# 3. Check if you have enough disk space for model cache

# If queries are slow:
# 1. Enable caching: ENABLE_LLM_CACHE = True
# 2. Reduce max_new_tokens
# 3. Use fewer contexts for testing
