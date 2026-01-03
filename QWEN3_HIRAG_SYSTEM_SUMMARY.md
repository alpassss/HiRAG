# Qwen3-8B HiRAG System for Kaggle Notebook - Complete Setup

## Overview
This system implements a complete HiRAG (Hierarchical Retrieval-Augmented Generation) system using:
- **LLM**: Qwen/Qwen3-8B (base model, not instruct)
- **Embedding Model**: sentence-transformers/all-mpnet-base-v2 (optimized for H100)
- **Dataset**: eval/datasets/cs/cs_unique_contexts.json
- **Target**: H100 GPU (80GB) in notebook environment

## Files in the System

### Core System Files
1. **qwen3_notebook_system.py** - Main HiRAG system implementation optimized for notebook environments
2. **kaggle_notebook_qwen3_hirag.ipynb** - Jupyter notebook interface for easy use
3. **run_qwen3_hirag_notebook.py** - Python script version for notebook execution

### Key Features
- ✅ Uses Qwen3-8B base model (not instruct) as requested
- ✅ Uses all-mpnet-base-v2 embedding model optimized for H100
- ✅ Progress bars for all operations (data loading, insertion, querying)
- ✅ Timing for each step
- ✅ Asynchronous operations with notebook compatibility (via nest_asyncio)
- ✅ Processes knowledge graph using eval/datasets/cs/cs_unique_contexts.json
- ✅ Optimized for H100 GPU (80GB) with bfloat16 precision
- ✅ Easy parameter modification in both notebook and script versions

## Installation & Setup

### Requirements
The system requires the following dependencies (already in requirements.txt):
```
sentence-transformers==3.2.1
transformers==4.47.1
torch
tqdm
tiktoken
nest_asyncio
```

### Installation Commands
```bash
pip install -r requirements.txt
```

## Usage Instructions

### Option 1: Jupyter Notebook (Recommended for Kaggle)
```python
# In the notebook:
from qwen3_notebook_system import run_notebook_pipeline

results = run_notebook_pipeline(
    llm_model_name="Qwen/Qwen3-8B",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    data_file="/workspace/eval/datasets/cs/cs_unique_contexts.json",
    num_contexts=10,
    queries=[
        "Your query here",
        "Another query",
        "Third query"
    ]
)
```

### Option 2: Python Script
```bash
python run_qwen3_hirag_notebook.py
```

### Option 3: Direct Function Call
```python
from run_qwen3_hirag_notebook import run_custom_pipeline

results = run_custom_pipeline(
    llm_model_name="Qwen/Qwen3-8B",
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    data_file="/workspace/eval/datasets/cs/cs_unique_contexts.json",
    num_contexts=5,
    queries=["Your queries here"]
)
```

## Parameters That Can Be Easily Modified

### Model Names
- `llm_model_name`: Default "Qwen/Qwen3-8B"
- `embedding_model_name`: Default "sentence-transformers/all-mpnet-base-v2"

### Data Parameters
- `data_file`: Path to JSON data file
- `num_contexts`: Number of contexts to process

### Query Parameters
- `queries`: List of queries to process

### HiRAG Configuration
The system uses a comprehensive configuration class that allows modification of:
- Chunk sizes and overlap
- Graph clustering parameters
- Node embedding parameters
- LLM settings
- And many more

## GPU Optimization Features

### Memory Management
- Uses bfloat16 precision for Qwen3-8B (optimal for H100)
- Automatic device mapping to GPU
- Memory cleanup functions
- Batch processing for embeddings

### Performance Optimizations
- Async operations with controlled concurrency
- Progress tracking for long operations
- Efficient data loading and processing
- Optimized embedding batch sizes

## Verification Steps

### 1. Check GPU Availability
```python
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 2. Test Import
```python
from qwen3_notebook_system import Qwen3HiRAG, Qwen3HiRAGConfig
print("System imported successfully!")
```

### 3. Quick Test
```python
from qwen3_notebook_system import run_notebook_pipeline

# Quick test with minimal data
results = run_notebook_pipeline(
    num_contexts=2,  # Small number for quick test
    queries=["What is machine learning?"]
)
```

## Troubleshooting

### Common Issues and Solutions:

1. **GPU Memory Issues**:
   - Reduce `num_contexts` parameter
   - Reduce batch sizes in configuration
   - Clear GPU cache with `torch.cuda.empty_cache()`

2. **Model Loading Issues**:
   - Ensure Hugging Face token is set if using gated models
   - Check internet connectivity for model downloads
   - Verify model names are correct

3. **Async Issues in Notebook**:
   - Make sure `nest_asyncio.apply()` is called
   - Restart kernel if needed

4. **Import Errors**:
   - Verify all dependencies are installed
   - Check file paths are correct
   - Ensure all required files exist

## Performance Expectations

With H100 (80GB) GPU:
- Model loading: 2-5 minutes
- Data processing: 1-3 minutes per 10 contexts
- Query processing: 30-90 seconds per query depending on complexity
- Total pipeline time: 5-15 minutes for full execution

## Data Flow

1. Load JSON data from `/workspace/eval/datasets/cs/cs_unique_contexts.json`
2. Initialize Qwen3-8B and embedding models on GPU
3. Process contexts through HiRAG knowledge graph
4. Execute queries against the knowledge graph
5. Return results with timing information

## Architecture

```
[JSON Data] 
    ↓
[Qwen3-8B LLM + all-mpnet-base-v2 Embedding Model on H100 GPU]
    ↓
[HiRAG Knowledge Graph Construction]
    ↓
[Query Processing with Progress Bars and Timing]
    ↓
[Results Output]
```

This system is fully optimized for the Kaggle notebook environment with H100 GPU, providing an efficient and user-friendly interface for HiRAG operations with Qwen3-8B.