# HiRAG with Qwen3 Models - Kaggle Notebook Usage Guide

This guide explains how to use the `notebook_hirag_qwen3.py` script in Kaggle Notebook environment with Qwen3 models from Hugging Face.

## Overview

This implementation uses:
- **LLM**: Qwen/Qwen3-8B-Base (non-instruct version) from Hugging Face
- **Embedding Model**: Qwen/Qwen3-Embedding-0.6B from Hugging Face
- **GPU**: Optimized for H100 80GB (but works on other GPUs)
- **Dataset**: eval/datasets/cs/cs_unique_contexts.json

## Features

✅ Async execution with notebook compatibility (nest_asyncio)
✅ Progress bars for all major operations
✅ Timing information for each step
✅ Easily configurable parameters
✅ GPU support with automatic device detection
✅ Memory-efficient batch processing
✅ LLM response caching

## Installation

### In Kaggle Notebook:

```python
# Install the required dependencies
!pip install -e .

# Or install individual packages if needed:
!pip install torch transformers accelerate sentencepiece nest_asyncio tqdm
```

### GPU Configuration

The script automatically detects and uses GPU if available. For H100 80GB, it uses:
- `torch.bfloat16` for optimal performance
- Automatic device mapping for efficient memory usage

## Usage

### Basic Usage

1. **Upload the repository to Kaggle** or clone it:
   ```bash
   !git clone https://github.com/alpassss/HiRAG.git
   %cd HiRAG
   ```

2. **Run the script**:
   ```python
   %run notebook_hirag_qwen3.py
   ```

### Configuration

All parameters are configurable at the top of the `notebook_hirag_qwen3.py` file:

#### File Paths
```python
DATA_FILE = "./eval/datasets/cs/cs_unique_contexts.json"  # Change to your data file
WORKING_DIR = "./hirag_qwen3_workdir"  # Change to your working directory
```

#### Model Configuration
```python
LLM_MODEL_NAME = "Qwen/Qwen3-8B-Base"  # LLM model name
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # Embedding model name
```

#### Model Parameters
```python
EMBEDDING_DIM = 1024  # Embedding dimension
MAX_TOKEN_SIZE = 8192  # Maximum token size for embeddings
EMBEDDING_BATCH_SIZE = 16  # Batch size for embedding computation
MAX_NEW_TOKENS = 2048  # Maximum tokens to generate
```

#### HiRAG Configuration
```python
ENABLE_LLM_CACHE = True  # Enable LLM response caching
ENABLE_HIERARCHICAL_MODE = True  # Enable hierarchical mode
ENABLE_NAIVE_RAG = True  # Enable naive RAG mode
EMBEDDING_BATCH_NUM = 6
EMBEDDING_FUNC_MAX_ASYNC = 8
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100
```

#### Data Loading
```python
NUM_CONTEXTS_TO_LOAD = None  # None = all, or set to 100 for testing
```

#### Query Configuration
```python
TEST_QUERY = "What are the main concepts discussed in computer science?"
QUERY_MODE = "hi"  # Options: "hi", "naive", "hi_nobridge", "hi_local", "hi_global", "hi_bridge"
```

### Advanced Usage

#### Performing Additional Queries

After running the script, you can perform additional queries:

```python
# Hierarchical query (recommended)
response = graph_func.query(
    "Your question here", 
    param=QueryParam(mode="hi")
)
print(response)

# Naive RAG query
response = graph_func.query(
    "Your question here", 
    param=QueryParam(mode="naive")
)
print(response)

# Local knowledge only
response = graph_func.query(
    "Your question here", 
    param=QueryParam(mode="hi_local")
)
print(response)

# Global knowledge only
response = graph_func.query(
    "Your question here", 
    param=QueryParam(mode="hi_global")
)
print(response)
```

#### Using Different Datasets

To use a different dataset:

1. Change the `DATA_FILE` parameter:
   ```python
   DATA_FILE = "./path/to/your/data.json"
   ```

2. Ensure your JSON file is in the correct format:
   - A list of strings (text contexts)
   ```json
   ["context 1", "context 2", "context 3", ...]
   ```

#### Testing with Smaller Dataset

For quick testing, limit the number of contexts:

```python
NUM_CONTEXTS_TO_LOAD = 50  # Load only first 50 contexts
```

## Query Modes

The system supports multiple query modes:

- **`hi`**: Full hierarchical query with local, global, and bridge knowledge
- **`naive`**: Simple RAG with text chunks only
- **`hi_nobridge`**: Hierarchical query without bridge knowledge
- **`hi_local`**: Only local knowledge (entities)
- **`hi_global`**: Only global knowledge (community reports)
- **`hi_bridge`**: Only bridge knowledge (relationships)

## Performance Monitoring

The script provides:

1. **Progress bars** for major operations:
   - Model loading
   - Data loading
   - Knowledge graph construction
   - Query execution

2. **Timing information** for each step:
   - ⏱️ Starting: [operation]
   - ✅ Completed: [operation] (Time: X.XXs)

3. **GPU memory usage** (if using CUDA):
   - Memory allocated
   - Memory reserved

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce batch sizes:
   ```python
   EMBEDDING_BATCH_SIZE = 8  # Reduce from 16
   EMBEDDING_BATCH_NUM = 4  # Reduce from 6
   ```

2. Reduce context length:
   ```python
   CHUNK_TOKEN_SIZE = 800  # Reduce from 1200
   ```

3. Use fewer contexts for testing:
   ```python
   NUM_CONTEXTS_TO_LOAD = 50  # Start small
   ```

### Async Issues in Notebook

The script uses `nest_asyncio` to handle async operations in Jupyter/Kaggle notebooks. If you encounter async-related errors:

```python
import nest_asyncio
nest_asyncio.apply()
```

### Slow Model Loading

First-time model loading downloads models from Hugging Face. This can take several minutes:
- Qwen3-8B-Base: ~16GB
- Qwen3-Embedding-0.6B: ~2.4GB

Models are cached after the first download.

## Example Notebook Workflow

```python
# 1. Install dependencies
!pip install -e .

# 2. Run the main script (builds knowledge graph and performs test query)
%run notebook_hirag_qwen3.py

# 3. Perform additional queries
response = graph_func.query(
    "What are the key algorithms discussed?",
    param=QueryParam(mode="hi")
)
print(response)

# 4. Try different query modes
response_naive = graph_func.query(
    "Explain the main topics",
    param=QueryParam(mode="naive")
)
print(response_naive)

# 5. Check system status
print(f"Working directory: {WORKING_DIR}")
print(f"Device: {DEVICE}")
print(f"Contexts loaded: {len(contexts)}")
```

## GPU Requirements

- **Minimum**: 16GB VRAM (for basic operation)
- **Recommended**: 40GB+ VRAM (for optimal performance)
- **Tested on**: H100 80GB

For lower VRAM GPUs:
- Use smaller models (e.g., Qwen2.5-3B)
- Reduce batch sizes
- Enable model quantization

## References

- Qwen3 Models: https://github.com/QwenLM/Qwen3
- Qwen3-Embedding: https://qwenlm.github.io/blog/qwen3-embedding/
- HiRAG Paper: https://arxiv.org/abs/2503.10150

## Support

For issues or questions:
- GitHub Issues: https://github.com/alpassss/HiRAG/issues
- HiRAG Documentation: See main README.md
