# Qwen3 Integration for HiRAG

## Quick Start with Qwen3 Models

### For Kaggle Notebook Users

If you want to use HiRAG with Qwen3 models from Hugging Face in a Kaggle notebook environment:

```python
# 1. Install HiRAG
!pip install -e .

# 2. Run the notebook script
%run notebook_hirag_qwen3.py
```

That's it! The script will:
- Load Qwen3-8B-Base and Qwen3-Embedding-0.6B models
- Build the knowledge graph from CS dataset
- Execute a test query
- Display progress and timing for all steps

### Models Used

- **LLM**: [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) - Non-instruct version
- **Embedding**: [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) - 1024-dim embeddings

### Key Features

✅ **GPU Optimized**: Automatic device detection, optimized for H100 80GB
✅ **Notebook Compatible**: Uses nest_asyncio for seamless async in notebooks
✅ **Progress Tracking**: Progress bars and timing for all operations
✅ **Easy Configuration**: All parameters editable at top of script
✅ **Memory Efficient**: Batch processing and GPU cache management
✅ **Error Handling**: Comprehensive error handling with fallbacks

## Configuration

Edit these parameters at the top of `notebook_hirag_qwen3.py`:

```python
# Data source
DATA_FILE = "./eval/datasets/cs/cs_unique_contexts.json"

# Number of contexts (None = all, or set to 50 for testing)
NUM_CONTEXTS_TO_LOAD = None

# Model settings
EMBEDDING_BATCH_SIZE = 16  # Reduce if OOM
MAX_NEW_TOKENS = 2048

# Query settings
TEST_QUERY = "Your question here"
QUERY_MODE = "hi"  # "hi", "naive", "hi_local", "hi_global", "hi_bridge"
```

## Usage Examples

### Basic Query
```python
response = graph_func.query(
    "What are the main concepts?",
    param=QueryParam(mode="hi")
)
print(response)
```

### Compare Query Modes
```python
modes = ["hi", "naive", "hi_local", "hi_global"]
for mode in modes:
    response = graph_func.query("Your question", param=QueryParam(mode=mode))
    print(f"\n{mode}: {response}")
```

### Use Different Dataset
```python
# Edit in notebook_hirag_qwen3.py:
DATA_FILE = "./path/to/your/data.json"

# Then run:
%run notebook_hirag_qwen3.py
```

## Documentation

- **English Guide**: [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)
- **中文指南**: [NOTEBOOK_USAGE_CN.md](NOTEBOOK_USAGE_CN.md)
- **Examples**: [notebook_example_usage.py](notebook_example_usage.py)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## Requirements

### Hardware
- **Minimum**: 16GB GPU VRAM
- **Recommended**: 40GB+ GPU VRAM
- **Tested on**: H100 80GB

### Software
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.40+
- CUDA (for GPU support)

All dependencies are listed in `requirements.txt`.

## Troubleshooting

### Out of Memory (OOM)
Reduce batch sizes in the configuration:
```python
EMBEDDING_BATCH_SIZE = 8  # Default: 16
NUM_CONTEXTS_TO_LOAD = 50  # Test with fewer contexts
```

### Models Not Loading
First run downloads models (~18GB). Ensure:
- Stable internet connection
- Sufficient disk space
- Access to Hugging Face hub

### Slow Performance
- Enable caching: `ENABLE_LLM_CACHE = True`
- Use fewer contexts for testing
- Reduce `MAX_NEW_TOKENS`

## Performance Notes

- **First Run**: Slower (downloads models + builds graph)
- **Subsequent Runs**: Faster (uses cached models + responses)
- **Query Time**: Varies by mode, typically 2-10 seconds

## Advantages

Compared to API-based approaches:
- ✅ No API costs
- ✅ No rate limits
- ✅ Full control over models
- ✅ Better privacy (data stays local)
- ✅ Reproducible results

## Support

Need help?
1. Check [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md) for detailed instructions
2. See [notebook_example_usage.py](notebook_example_usage.py) for examples
3. Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical details
4. Open an issue on GitHub

## Credits

This implementation uses:
- [Qwen3 Models](https://github.com/QwenLM/Qwen3) by Alibaba
- [HiRAG](https://arxiv.org/abs/2503.10150) framework
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
