# HiRAG Qwen3模型 - Kaggle Notebook快速使用指南

## 概述

本实现使用以下模型：
- **语言模型**: Qwen/Qwen3-8B-Base（非instruct版本）来自Hugging Face
- **嵌入模型**: Qwen/Qwen3-Embedding-0.6B 来自Hugging Face  
- **GPU**: 针对H100 80GB优化（也支持其他GPU）
- **数据集**: eval/datasets/cs/cs_unique_contexts.json

## 主要特性

✅ 支持Notebook环境的异步执行（使用nest_asyncio）
✅ 所有主要操作都有进度条显示
✅ 每个步骤都有计时信息
✅ 参数可以方便地配置
✅ 自动GPU检测和支持
✅ 内存高效的批处理
✅ LLM响应缓存

## 安装

### 在Kaggle Notebook中：

```python
# 安装所需依赖
!pip install -e .

# 或单独安装所需包:
!pip install torch transformers accelerate sentencepiece nest_asyncio tqdm
```

## 使用方法

### 基础使用

1. **上传仓库到Kaggle**或克隆：
   ```bash
   !git clone https://github.com/alpassss/HiRAG.git
   %cd HiRAG
   ```

2. **运行脚本**：
   ```python
   %run notebook_hirag_qwen3.py
   ```

### 配置参数

所有参数都可以在`notebook_hirag_qwen3.py`文件顶部配置：

#### 文件路径
```python
DATA_FILE = "./eval/datasets/cs/cs_unique_contexts.json"  # 修改为你的数据文件
WORKING_DIR = "./hirag_qwen3_workdir"  # 修改为你的工作目录
```

#### 模型配置
```python
LLM_MODEL_NAME = "Qwen/Qwen3-8B-Base"  # 语言模型名称
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # 嵌入模型名称
```

#### 模型参数
```python
EMBEDDING_DIM = 1024  # 嵌入维度
MAX_TOKEN_SIZE = 8192  # 嵌入的最大token大小
EMBEDDING_BATCH_SIZE = 16  # 嵌入计算的批次大小
MAX_NEW_TOKENS = 2048  # 生成的最大token数
```

#### HiRAG配置
```python
ENABLE_LLM_CACHE = True  # 启用LLM响应缓存
ENABLE_HIERARCHICAL_MODE = True  # 启用分层模式
ENABLE_NAIVE_RAG = True  # 启用简单RAG模式
EMBEDDING_BATCH_NUM = 6
EMBEDDING_FUNC_MAX_ASYNC = 8
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100
```

#### 数据加载
```python
NUM_CONTEXTS_TO_LOAD = None  # None = 加载全部，或设置为100进行测试
```

#### 查询配置
```python
TEST_QUERY = "计算机科学中讨论的主要概念是什么？"
QUERY_MODE = "hi"  # 选项: "hi", "naive", "hi_nobridge", "hi_local", "hi_global", "hi_bridge"
```

### 高级使用

#### 执行额外的查询

运行脚本后，你可以执行额外的查询：

```python
# 分层查询（推荐）
response = graph_func.query(
    "你的问题", 
    param=QueryParam(mode="hi")
)
print(response)

# 简单RAG查询
response = graph_func.query(
    "你的问题", 
    param=QueryParam(mode="naive")
)
print(response)

# 仅使用本地知识
response = graph_func.query(
    "你的问题", 
    param=QueryParam(mode="hi_local")
)
print(response)

# 仅使用全局知识
response = graph_func.query(
    "你的问题", 
    param=QueryParam(mode="hi_global")
)
print(response)
```

#### 使用不同的数据集

要使用不同的数据集：

1. 修改`DATA_FILE`参数：
   ```python
   DATA_FILE = "./path/to/your/data.json"
   ```

2. 确保你的JSON文件格式正确：
   - 一个字符串列表（文本上下文）
   ```json
   ["上下文1", "上下文2", "上下文3", ...]
   ```

#### 使用较小数据集测试

快速测试时，限制上下文数量：

```python
NUM_CONTEXTS_TO_LOAD = 50  # 仅加载前50个上下文
```

## 查询模式

系统支持多种查询模式：

- **`hi`**: 完整的分层查询，包含本地、全局和桥接知识
- **`naive`**: 简单的RAG，仅使用文本块
- **`hi_nobridge`**: 不包含桥接知识的分层查询
- **`hi_local`**: 仅使用本地知识（实体）
- **`hi_global`**: 仅使用全局知识（社区报告）
- **`hi_bridge`**: 仅使用桥接知识（关系）

## 性能监控

脚本提供以下功能：

1. **主要操作的进度条**：
   - 模型加载
   - 数据加载
   - 知识图谱构建
   - 查询执行

2. **每个步骤的计时信息**：
   - ⏱️ 开始: [操作]
   - ✅ 完成: [操作] (时间: X.XXs)

3. **GPU内存使用情况**（如果使用CUDA）：
   - 已分配内存
   - 保留内存

## 故障排除

### 内存不足（OOM）

如果遇到OOM错误：

1. 减少批次大小：
   ```python
   EMBEDDING_BATCH_SIZE = 8  # 从16减少
   EMBEDDING_BATCH_NUM = 4  # 从6减少
   ```

2. 减少上下文长度：
   ```python
   CHUNK_TOKEN_SIZE = 800  # 从1200减少
   ```

3. 使用更少的上下文进行测试：
   ```python
   NUM_CONTEXTS_TO_LOAD = 50  # 从小开始
   ```

### Notebook中的异步问题

脚本使用`nest_asyncio`处理Jupyter/Kaggle notebook中的异步操作。如果遇到异步相关错误：

```python
import nest_asyncio
nest_asyncio.apply()
```

### BFloat16/Float16转换错误

如果遇到"Got unsupported ScalarType BFloat16"错误，此问题已在最新版本中修复。嵌入函数现在会在numpy转换前自动将bfloat16/float16张量转换为float32。

如果使用旧版本，请更新嵌入函数以包含：
```python
# 如果使用bfloat16或float16，转换为float32
if embeddings.dtype in (torch.bfloat16, torch.float16):
    embeddings = embeddings.to(torch.float32)
```

### 模型名称混淆

确保使用正确的模型名称：
- ✅ `Qwen/Qwen3-8B-Base` - 非instruct版本（推荐）
- ❌ `Qwen/Qwen3-8B` - 不同的模型，可能无法正常工作

### 模型加载缓慢

首次加载模型会从Hugging Face下载模型。这可能需要几分钟：
- Qwen3-8B-Base: ~16GB
- Qwen3-Embedding-0.6B: ~2.4GB

首次下载后模型会被缓存。

## Notebook工作流示例

```python
# 1. 安装依赖
!pip install -e .

# 2. 运行主脚本（构建知识图谱并执行测试查询）
%run notebook_hirag_qwen3.py

# 3. 执行额外查询
response = graph_func.query(
    "讨论了哪些关键算法？",
    param=QueryParam(mode="hi")
)
print(response)

# 4. 尝试不同的查询模式
response_naive = graph_func.query(
    "解释主要主题",
    param=QueryParam(mode="naive")
)
print(response_naive)

# 5. 检查系统状态
print(f"工作目录: {WORKING_DIR}")
print(f"设备: {DEVICE}")
print(f"加载的上下文: {len(contexts)}")
```

## GPU要求

- **最低**: 16GB VRAM（基本操作）
- **推荐**: 40GB+ VRAM（最佳性能）
- **已测试**: H100 80GB

对于较低VRAM的GPU：
- 使用较小的模型（如Qwen2.5-3B）
- 减少批次大小
- 启用模型量化

## 参考资料

- Qwen3模型: https://github.com/QwenLM/Qwen3
- Qwen3-Embedding: https://qwenlm.github.io/blog/qwen3-embedding/
- HiRAG论文: https://arxiv.org/abs/2503.10150

## 支持

如有问题：
- GitHub Issues: https://github.com/alpassss/HiRAG/issues
- HiRAG文档: 参见主README.md

## 完整示例代码

参见`notebook_example_usage.py`文件，了解更多使用示例。
