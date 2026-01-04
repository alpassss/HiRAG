# Implementation Summary: HiRAG with Qwen3 Models

## Overview

This implementation converts the HiRAG system to use Hugging Face models instead of API calls, specifically:
- **LLM Model**: Qwen/Qwen3-8B-Base (non-instruct version)
- **Embedding Model**: Qwen/Qwen3-Embedding-0.6B
- **Target Environment**: Kaggle Notebook with H100 80GB GPU

## Files Created

### 1. `notebook_hirag_qwen3.py`
The main script that integrates Qwen3 models with HiRAG for use in Kaggle notebooks.

**Key Features:**
- Automatic GPU detection and configuration
- Support for bfloat16 on H100/A100, float16 on older GPUs
- Async execution with nest_asyncio for notebook compatibility
- Progress bars for all major operations
- Timing information for performance monitoring
- Memory-efficient batch processing
- GPU cache management to prevent OOM
- Comprehensive error handling with fallback responses
- LLM response caching for efficiency

**Configuration Options (all at top of file):**
```python
# File paths
DATA_FILE = "./eval/datasets/cs/cs_unique_contexts.json"
WORKING_DIR = "./hirag_qwen3_workdir"

# Models
LLM_MODEL_NAME = "Qwen/Qwen3-8B-Base"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Parameters
EMBEDDING_DIM = 1024
MAX_TOKEN_SIZE = 8192
EMBEDDING_BATCH_SIZE = 16
MAX_NEW_TOKENS = 2048
NUM_CONTEXTS_TO_LOAD = None  # None = all, or set to number for testing

# HiRAG settings
ENABLE_LLM_CACHE = True
ENABLE_HIERARCHICAL_MODE = True
ENABLE_NAIVE_RAG = True
EMBEDDING_BATCH_NUM = 6
EMBEDDING_FUNC_MAX_ASYNC = 8
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100

# Query settings
TEST_QUERY = "What are the main concepts discussed in computer science?"
QUERY_MODE = "hi"  # Options: "hi", "naive", "hi_nobridge", "hi_local", "hi_global", "hi_bridge"
```

**Key Functions:**

1. `qwen3_embedding(texts)` - Async embedding function
   - Batch processing for memory efficiency
   - Mean pooling on last hidden state
   - Automatic GPU cache clearing
   - Error handling

2. `qwen3_llm_if_cache(prompt, ...)` - Async LLM function
   - Caching support
   - Flexible prompt formatting
   - Temperature and top_p control
   - Error handling with fallback
   - GPU cache management

3. `Timer` class - Context manager for timing operations

### 2. `NOTEBOOK_USAGE.md`
Comprehensive English documentation covering:
- Installation instructions
- Usage examples
- Configuration options
- Query modes explanation
- Performance monitoring
- Troubleshooting guide
- GPU requirements
- Example workflows

### 3. `NOTEBOOK_USAGE_CN.md`
Chinese version of the documentation for Chinese-speaking users.

### 4. `notebook_example_usage.py`
Practical examples showing:
- Basic query execution
- Batch query processing
- Comparing different query modes
- GPU memory monitoring
- Configuration modification
- Cleanup procedures
- Troubleshooting tips

### 5. `requirements.txt` (updated)
Added necessary dependencies:
```
torch>=2.0.0
accelerate>=0.20.0
sentencepiece>=0.1.99
```

## Technical Details

### Model Integration

#### Embedding Model
- Uses Qwen/Qwen3-Embedding-0.6B
- Output dimension: 1024 (configurable 32-1024 with Matryoshka)
- Supports 100+ languages including Chinese and English
- Mean pooling strategy for sentence embeddings
- Batch processing with configurable batch size

#### LLM Model
- Uses Qwen/Qwen3-8B-Base (non-instruct version)
- Causal language model without instruction tuning
- Pipeline-based generation for efficiency
- Configurable generation parameters (temperature, top_p, max_tokens)
- Prompt formatting for conversation-style interaction

### GPU Optimization

1. **Automatic dtype selection**:
   - bfloat16 for compute capability >= 8.0 (H100, A100)
   - float16 for older GPUs

2. **Memory management**:
   - Batch processing for embeddings
   - Periodic GPU cache clearing
   - Automatic device mapping
   - Memory monitoring and reporting

3. **Error handling**:
   - Try-catch blocks for model operations
   - Fallback responses on errors
   - Comprehensive logging

### Async Handling

- Uses `nest_asyncio` to enable nested event loops in Jupyter/Kaggle
- Compatible with HiRAG's async architecture
- Maintains async efficiency while working in notebook environments

### Progress Tracking

- Timer class for measuring execution time
- Progress indicators for major operations:
  - Model loading
  - Data loading
  - Knowledge graph construction
  - Query execution
- GPU memory usage reporting

## Usage Flow

1. **Initialization**:
   ```python
   %run notebook_hirag_qwen3.py
   ```
   - Loads models from Hugging Face
   - Initializes HiRAG system
   - Loads data from JSON file
   - Builds knowledge graph
   - Executes test query

2. **Additional Queries**:
   ```python
   response = graph_func.query("Your question", param=QueryParam(mode="hi"))
   ```

3. **Configuration Changes**:
   - Edit parameters at top of `notebook_hirag_qwen3.py`
   - Re-run the script

## Advantages Over API-based Approach

1. **No API costs**: Run locally on GPU without per-request charges
2. **No rate limits**: No API rate limiting issues
3. **Full control**: Complete control over model parameters and behavior
4. **Privacy**: Data stays local, not sent to external APIs
5. **Reproducibility**: Consistent results with fixed model versions
6. **Customization**: Can modify models or use different ones easily

## Performance Considerations

### Memory Requirements
- **Minimum**: ~24GB (8B LLM + 0.6B embedding + overhead)
- **Recommended**: 40GB+ for comfortable operation
- **Tested**: H100 80GB (optimal)

### Speed
- First run: Slower (model download + graph construction)
- Subsequent runs: Faster (cached models + LLM response cache)
- Query time: Depends on complexity and caching

### Optimization Tips
1. Use smaller context windows for testing
2. Enable LLM caching
3. Reduce batch sizes if OOM occurs
4. Use fewer contexts initially
5. Monitor GPU memory usage

## Query Modes Supported

All HiRAG query modes are fully supported:

- **hi**: Full hierarchical retrieval (recommended)
- **naive**: Simple RAG with text chunks only
- **hi_nobridge**: Hierarchical without bridge knowledge
- **hi_local**: Only local entity knowledge
- **hi_global**: Only global community knowledge
- **hi_bridge**: Only bridge relationship knowledge

## Data Format

Expected JSON format:
```json
["context string 1", "context string 2", "context string 3", ...]
```

Default dataset: `eval/datasets/cs/cs_unique_contexts.json`

## Future Enhancements

Possible improvements:
1. Model quantization (4-bit, 8-bit) for lower memory usage
2. Support for other Qwen model variants
3. Multi-GPU support for parallel processing
4. Streaming responses for better UX
5. Fine-tuning support for domain-specific tasks
6. Integration with other embedding models
7. Automatic model selection based on available GPU memory

## Testing Recommendations

1. Start with small dataset (NUM_CONTEXTS_TO_LOAD = 50)
2. Verify GPU detection and memory availability
3. Test basic query execution
4. Try different query modes
5. Monitor memory usage throughout
6. Scale up to full dataset once validated

## Troubleshooting Guide

### Issue: OOM Error
**Solutions**:
- Reduce EMBEDDING_BATCH_SIZE
- Reduce NUM_CONTEXTS_TO_LOAD
- Reduce CHUNK_TOKEN_SIZE
- Use model quantization

### Issue: Slow Generation
**Solutions**:
- Reduce MAX_NEW_TOKENS
- Enable caching (ENABLE_LLM_CACHE = True)
- Use fewer contexts for testing

### Issue: Model Download Fails
**Solutions**:
- Check internet connection
- Verify Hugging Face hub accessibility
- Check disk space
- Try manual download

### Issue: Async Errors
**Solutions**:
- Ensure nest_asyncio is installed
- Verify nest_asyncio.apply() is called
- Check Python version (3.8+)

## References

- Qwen3 Models: https://github.com/QwenLM/Qwen3
- Qwen3-Embedding: https://qwenlm.github.io/blog/qwen3-embedding/
- HiRAG Paper: https://arxiv.org/abs/2503.10150
- Transformers Library: https://huggingface.co/docs/transformers

## Support

For issues or questions:
- Check NOTEBOOK_USAGE.md for detailed instructions
- Review notebook_example_usage.py for examples
- See main repository README.md for HiRAG documentation
- Open GitHub issue if problems persist
