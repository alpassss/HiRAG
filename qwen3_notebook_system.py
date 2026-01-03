"""
HiRAG System with Qwen3 8B for Notebook Environment
This script provides a complete system for using Qwen3 8B model and embedding models from Hugging Face
in the HiRAG framework. It includes progress bars and timing for each step, with lazy model loading
to work in notebook environments.
"""
import asyncio
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Callable
from dataclasses import dataclass, field
from tqdm.asyncio import tqdm
from tqdm import tqdm as tqdm_sync
import tiktoken

# Import necessary HiRAG components (deferred until actually needed)


@dataclass
class Qwen3HiRAGConfig:
    """
    Configuration class for Qwen3 HiRAG system
    All parameters can be easily modified for different configurations
    """
    # Model configurations
    llm_model_name: str = "Qwen/Qwen3-8B"  # Using the base Qwen3 8B model (not instruct)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Suitable embedding model
    embedding_dimension: int = 384  # Dimension of the embedding model (all-MiniLM-L6-v2 has 384)
    
    # HiRAG configurations
    working_dir: str = "./hirag_cache_qwen3_notebook"
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_hierachical_mode: bool = True
    
    # Text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    
    # Entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500
    
    # Graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
    
    # Node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )
    
    # Text embedding
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 8
    query_better_than_threshold: float = 0.2
    
    # LLM settings
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 8
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 8
    
    # Community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    
    # Storage
    enable_llm_cache: bool = True
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)


class Qwen3HiRAG:
    """
    HiRAG system using Qwen3 8B model and Hugging Face embedding models
    With lazy loading to work in notebook environments
    """
    def __init__(self, config: 'Qwen3HiRAGConfig' = None):
        self.config = config or Qwen3HiRAGConfig()
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.hirag_instance = None
        
    def _init_models(self):
        """Initialize Qwen3 model and embedding model"""
        print("Initializing Qwen3 8B model and embedding model...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from sentence_transformers import SentenceTransformer
        
        start_time = time.time()
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use CPU for compatibility in notebook environments
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu"  # Use CPU to avoid CUDA issues
        )
        model_init_time = time.time() - start_time
        print(f"Model initialization completed in {model_init_time:.2f} seconds")
        
        # Initialize embedding model
        start_time = time.time()
        self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
        embedding_init_time = time.time() - start_time
        print(f"Embedding model initialization completed in {embedding_init_time:.2f} seconds")
    
    def _create_qwen3_completion(self):
        """Create completion function for Qwen3 model"""
        import torch
        
        async def qwen3_complete_if_cache(
            prompt, system_prompt=None, history_messages=[], **kwargs
        ) -> str:
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            
            # Tokenize the input
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([input_text], return_tensors="pt")
            
            # Move to model device if needed
            model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}
            
            # Generate response
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode the response
            decoded = self.tokenizer.batch_decode(
                generated_ids[:, model_inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )[0]
            
            return decoded
        
        return qwen3_complete_if_cache
    
    async def qwen3_embedding(self, texts: List[str]) -> np.ndarray:
        """Create embedding function for the embedding model"""
        # Process embeddings in batches to avoid memory issues
        all_embeddings = []
        
        for i in tqdm_sync(range(0, len(texts), self.config.embedding_batch_num), 
                          desc="Generating embeddings"):
            batch = texts[i:i + self.config.embedding_batch_num]
            # Encode the batch
            embeddings = self.embedding_model.encode(batch)
            all_embeddings.append(embeddings)
        
        # Concatenate all embeddings if there are any
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
            return final_embeddings.astype(np.float32)
        else:
            # Return empty array with correct shape if no texts
            return np.array([]).reshape(0, self.config.embedding_dimension).astype(np.float32)
    
    def get_embedding_func(self):
        """Get properly wrapped embedding function"""
        from hirag._utils import wrap_embedding_func_with_attrs
        return wrap_embedding_func_with_attrs(
            embedding_dim=self.config.embedding_dimension, 
            max_token_size=512
        )(self.qwen3_embedding)
    
    def create_hirag_instance(self):
        """Create HiRAG instance with Qwen3 models"""
        print("Creating HiRAG instance with Qwen3 models...")
        
        # Import HiRAG components when needed
        from hirag.hirag import HiRAG
        from hirag._utils import wrap_embedding_func_with_attrs
        
        # Initialize models if not already done
        if self.model is None or self.embedding_model is None:
            self._init_models()
        
        # Create LLM functions
        llm_func = self._create_qwen3_completion()
        
        # Initialize HiRAG with custom functions
        self.hirag_instance = HiRAG(
            working_dir=self.config.working_dir,
            enable_local=self.config.enable_local,
            enable_naive_rag=self.config.enable_naive_rag,
            enable_hierachical_mode=self.config.enable_hierachical_mode,
            chunk_token_size=self.config.chunk_token_size,
            chunk_overlap_token_size=self.config.chunk_overlap_token_size,
            entity_extract_max_gleaning=self.config.entity_extract_max_gleaning,
            entity_summary_to_max_tokens=self.config.entity_summary_to_max_tokens,
            graph_cluster_algorithm=self.config.graph_cluster_algorithm,
            max_graph_cluster_size=self.config.max_graph_cluster_size,
            graph_cluster_seed=self.config.graph_cluster_seed,
            node_embedding_algorithm=self.config.node_embedding_algorithm,
            node2vec_params=self.config.node2vec_params,
            embedding_batch_num=self.config.embedding_batch_num,
            embedding_func_max_async=self.config.embedding_func_max_async,
            query_better_than_threshold=self.config.query_better_than_threshold,
            best_model_max_token_size=self.config.best_model_max_token_size,
            best_model_max_async=self.config.best_model_max_async,
            cheap_model_max_token_size=self.config.cheap_model_max_async,
            cheap_model_max_async=self.config.cheap_model_max_async,
            enable_llm_cache=self.config.enable_llm_cache,
            always_create_working_dir=self.config.always_create_working_dir,
            addon_params=self.config.addon_params,
        )
        
        # Override the embedding and LLM functions
        self.hirag_instance.embedding_func = self.limit_async_func_call(
            self.config.embedding_func_max_async
        )(self.get_embedding_func())
        
        self.hirag_instance.best_model_func = self.limit_async_func_call(
            self.config.best_model_max_async
        )(llm_func)
        
        self.hirag_instance.cheap_model_func = self.limit_async_func_call(
            self.config.cheap_model_max_async
        )(llm_func)
        
        print("HiRAG instance created successfully")
        return self.hirag_instance
    
    def limit_async_func_call(self, max_size: int, waitting_time: float = 0.0001):
        """Add restriction of maximum async calling times for a async func"""
        def final_decro(func):
            __current_size = 0

            async def wait_func(*args, **kwargs):
                nonlocal __current_size
                while __current_size >= max_size:
                    await asyncio.sleep(waitting_time)
                __current_size += 1
                result = await func(*args, **kwargs)
                __current_size -= 1
                return result

            return wait_func

        return final_decro


async def load_cs_data(file_path: str) -> List[str]:
    """Load computer science contexts data"""
    print(f"Loading data from {file_path}...")
    start_time = time.time()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds. Total contexts: {len(data)}")
    return data


async def insert_data_with_progress(hirag_instance, contexts: List[str]):
    """Insert data into HiRAG with progress tracking"""
    print("Starting data insertion with progress tracking...")
    start_time = time.time()
    
    # Process contexts with progress bar
    for i, context in enumerate(tqdm_sync(contexts, desc="Inserting contexts")):
        await hirag_instance.ainsert([context])
        if (i + 1) % 10 == 0:  # Update every 10 contexts
            print(f"Inserted {i + 1}/{len(contexts)} contexts")
    
    total_time = time.time() - start_time
    print(f"Data insertion completed in {total_time:.2f} seconds")


async def query_with_progress(hirag_instance, query_text: str, query_param=None):
    """Query HiRAG with progress tracking"""
    # Import QueryParam only when needed
    from hirag.base import QueryParam
    
    print(f"Processing query: {query_text[:50]}...")
    start_time = time.time()
    
    if query_param is None:
        query_param = QueryParam()
    
    result = await hirag_instance.aquery(query_text, query_param)
    
    total_time = time.time() - start_time
    print(f"Query completed in {total_time:.2f} seconds")
    return result


def run_notebook_pipeline(
    llm_model_name: str = "Qwen/Qwen3-8B",
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    data_file: str = "/workspace/eval/datasets/cs/cs_unique_contexts.json",
    num_contexts: int = 10,
    queries: List[str] = None
):
    """
    Function to run the complete pipeline in a Jupyter notebook environment
    with progress bars and timing for each step.
    """
    if queries is None:
        queries = [
            "What are the key concepts in machine learning with Spark?",
            "Explain the architecture of a machine learning system",
            "How does collaborative filtering work in recommendation engines?"
        ]
    
    print("Starting Qwen3 HiRAG System Pipeline...")
    
    # Configuration
    config = Qwen3HiRAGConfig(
        llm_model_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        working_dir="./hirag_cache_qwen3_notebook",
    )
    
    # Initialize system
    start_time = time.time()
    qwen3_hirag = Qwen3HiRAG(config)
    init_time = time.time() - start_time
    print(f"System initialization completed in {init_time:.2f} seconds")
    
    # Create HiRAG instance
    start_time = time.time()
    hirag_instance = qwen3_hirag.create_hirag_instance()
    creation_time = time.time() - start_time
    print(f"HiRAG instance creation completed in {creation_time:.2f} seconds")
    
    # Load data
    contexts = asyncio.run(load_cs_data(data_file))
    print(f"Loaded {len(contexts)} contexts from {data_file}")
    
    # Use only specified number of contexts
    contexts = contexts[:num_contexts]
    print(f"Using {len(contexts)} contexts for processing")
    
    # Insert data
    start_time = time.time()
    asyncio.run(insert_data_with_progress(hirag_instance, contexts))
    insertion_time = time.time() - start_time
    print(f"Data insertion completed in {insertion_time:.2f} seconds")
    
    # Process queries
    results = []
    for i, query in enumerate(queries):
        print(f"\n--- Processing Query {i+1}/{len(queries)}: {query[:50]}... ---")
        result = asyncio.run(query_with_progress(hirag_instance, query))
        results.append(result)
        print(f"Query {i+1} completed")
    
    print(f"\nPipeline execution completed!")
    print(f"Total contexts processed: {len(contexts)}")
    print(f"Total queries processed: {len(queries)}")
    
    return results


# For testing the import
def test_import():
    """Test that the module can be imported without errors"""
    print("Qwen3 HiRAG System module loaded successfully!")
    print("Available functions:")
    print("- run_notebook_pipeline(): Main pipeline function for notebook use")
    print("- Qwen3HiRAGConfig: Configuration class")
    print("- Qwen3HiRAG: Main system class")
    return True


if __name__ == "__main__":
    test_import()