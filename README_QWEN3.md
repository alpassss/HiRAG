# 🚀 HiRAG with Qwen3 Models - Quick Start

## 中文说明 (Chinese)

本实现将HiRAG系统配置为使用来自Hugging Face的Qwen3-8B-Base（非instruct版本）和Qwen3-Embedding-0.6B模型，专为Kaggle notebook环境和H100 80GB GPU优化。

### 快速开始

```python
# 在Kaggle notebook中运行
!pip install -e .
%run notebook_hirag_qwen3.py
```

详细中文文档请参见: [NOTEBOOK_USAGE_CN.md](NOTEBOOK_USAGE_CN.md)

---

## English

This implementation configures HiRAG to use **Qwen3-8B-Base** (non-instruct version) and **Qwen3-Embedding-0.6B** models from Hugging Face, optimized for Kaggle notebook environment with H100 80GB GPU.

### Quick Start

```python
# Run in Kaggle notebook
!pip install -e .
%run notebook_hirag_qwen3.py
```

Detailed English documentation: [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)

---

## ✨ What You Get

✅ **Qwen3-8B-Base** LLM (non-instruct version)  
✅ **Qwen3-Embedding-0.6B** for embeddings (1024-dim)  
✅ **GPU Support** - Auto-detection, optimized for H100  
✅ **Progress Bars** - Track every operation  
✅ **Async Compatible** - Works in Jupyter/Kaggle notebooks  
✅ **Easy Configuration** - All parameters at top of file  
✅ **Comprehensive Docs** - English & Chinese guides  

---

## 📁 Files

| File | Description |
|------|-------------|
| **notebook_hirag_qwen3.py** | Main script - run this in notebook |
| **QWEN3_INTEGRATION.md** | Quick start guide |
| **NOTEBOOK_USAGE.md** | Detailed usage guide (English) |
| **NOTEBOOK_USAGE_CN.md** | 详细使用指南（中文）|
| **notebook_example_usage.py** | Code examples |
| **IMPLEMENTATION_SUMMARY.md** | Technical details |
| **IMPLEMENTATION_COMPLETE.md** | Implementation summary |

---

## 🎯 Requirements Met

✅ Uses Qwen3-8B (non-instruct)  
✅ Uses compatible embedding model (Qwen3-Embedding-0.6B)  
✅ Loads from eval/datasets/cs/cs_unique_contexts.json  
✅ Single file for notebook execution  
✅ Async execution with nest_asyncio  
✅ Progress bars for all steps  
✅ Timing for all operations  
✅ Configurable parameters  
✅ GPU support (H100 optimized)  

---

## 📖 Documentation

- **Quick Start**: [QWEN3_INTEGRATION.md](QWEN3_INTEGRATION.md)
- **English Guide**: [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md)
- **中文指南**: [NOTEBOOK_USAGE_CN.md](NOTEBOOK_USAGE_CN.md)
- **Examples**: [notebook_example_usage.py](notebook_example_usage.py)
- **Technical Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Implementation**: [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

---

## 💻 Usage in Kaggle

### Step 1: Upload & Install
```bash
# Upload this repo to Kaggle
# Enable GPU in Settings → Accelerator
!pip install -e .
```

### Step 2: Run
```python
%run notebook_hirag_qwen3.py
```

### Step 3: Query
```python
response = graph_func.query(
    "Your question here",
    param=QueryParam(mode="hi")
)
print(response)
```

---

## ⚙️ Configuration

Edit at the top of `notebook_hirag_qwen3.py`:

```python
# Data file
DATA_FILE = "./eval/datasets/cs/cs_unique_contexts.json"

# For testing (use fewer contexts)
NUM_CONTEXTS_TO_LOAD = 50  # or None for all

# Query settings
TEST_QUERY = "Your question"
QUERY_MODE = "hi"  # or "naive", "hi_local", etc.
```

---

## 🔧 Query Modes

- `hi` - Full hierarchical (recommended)
- `naive` - Simple RAG  
- `hi_local` - Local entities only
- `hi_global` - Global communities only
- `hi_bridge` - Bridge relationships only
- `hi_nobridge` - No bridge knowledge

---

## 🎮 GPU Requirements

- **Minimum**: 16GB VRAM
- **Recommended**: 40GB+ VRAM
- **Tested**: H100 80GB

---

## 📊 Performance

- **First Run**: ~15-40 minutes (model download + graph build)
- **Subsequent Runs**: ~2-10 seconds per query (cached)
- **Models Size**: ~18GB total

---

## 🆘 Troubleshooting

### Out of Memory?
```python
EMBEDDING_BATCH_SIZE = 8
NUM_CONTEXTS_TO_LOAD = 50
```

### Slow?
```python
ENABLE_LLM_CACHE = True
NUM_CONTEXTS_TO_LOAD = 100
```

See [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md) for detailed troubleshooting.

---

## 🌟 Features

- **No API Costs** - Run locally
- **No Rate Limits** - Process unlimited data
- **Full Control** - Customize everything
- **Privacy** - Data stays local
- **GPU Optimized** - bfloat16 on H100/A100

---

## 📞 Support

Need help?
1. Check [NOTEBOOK_USAGE.md](NOTEBOOK_USAGE.md) or [NOTEBOOK_USAGE_CN.md](NOTEBOOK_USAGE_CN.md)
2. Review [notebook_example_usage.py](notebook_example_usage.py)
3. See [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
4. Open GitHub issue

---

## ✅ Ready to Use!

Everything is implemented and ready. Just run:

```python
%run notebook_hirag_qwen3.py
```

🎉 **Start building your knowledge graph!**
