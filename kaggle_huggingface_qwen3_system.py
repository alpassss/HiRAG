"""
Kaggle Notebook System for HiRAG with Hugging Face Qwen3 8B Model
This script provides a synchronous implementation for use in Kaggle Notebooks
with Hugging Face models instead of OpenAI APIs.
"""

import json
import time
import numpy as np
from typing import List, Dict, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
import logging
import os
import tiktoken
from tqdm import tqdm

# Import transformers and torch for Hugging Face models
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from torch.nn import functional as F

# Import other required packages
from hirag._utils import EmbeddingFunc, compute_mdhash_id, logger
from hirag._storage import JsonKVStorage, NanoVectorDBStorage, NetworkXStorage
from hirag.base import BaseGraphStorage, BaseKVStorage, BaseVectorStorage, StorageNameSpace, QueryParam
from hirag._op import chunking_by_token_size, get_chunks, extract_hierarchical_entities, extract_entities, generate_community_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

class HuggingFaceLLM:
    """Hugging Face LLM wrapper for synchronous operations"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # For H100 GPU with 80GB memory, we can use more precision
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better quality on H100
            device_map="auto",  # Use available GPUs automatically
            trust_remote_code=True  # Required for Qwen models
        )
        self.model.eval()
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, system_prompt: Optional[str] = None, 
                 max_new_tokens: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """Generate response synchronously"""
        # Build the full prompt with system message if provided
        if system_prompt:
            full_prompt = f"<|system|>{system_prompt}<|end|>\n<|user|>{prompt}<|end|>\n<|assistant|>"
        else:
            full_prompt = f"<|user|>{prompt}<|end|>\n<|assistant|>"
        
        # Tokenize the input
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True, 
                               max_length=8192).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response part
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        else:
            response = response[len(full_prompt):].strip()
        
        return response

class HuggingFaceEmbedding:
    """Hugging Face embedding wrapper for synchronous operations"""
    
    def __init__(self, model_name: str = "Alibaba-NLP/gte-large-en-v1.5", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        self.max_token_size = 8192  # Updated for better compatibility

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings synchronously"""
        embeddings = []
        
        for text in tqdm(texts, desc="Generating embeddings", leave=False):
            # Tokenize the text
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=self.max_token_size
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get sentence embedding
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings.append(embedding.flatten())
        
        return np.array(embeddings)

# Create global instances
hf_llm_instance = None
hf_embedding_instance = None

def get_hf_llm_instance(model_name: str = "Qwen/Qwen3-8B"):
    global hf_llm_instance
    if hf_llm_instance is None:
        hf_llm_instance = HuggingFaceLLM(model_name=model_name)
    return hf_llm_instance

def get_hf_embedding_instance(model_name: str = "Alibaba-NLP/gte-large-en-v1.5"):
    global hf_embedding_instance
    if hf_embedding_instance is None:
        hf_embedding_instance = HuggingFaceEmbedding(model_name=model_name)
    return hf_embedding_instance

# Define synchronous LLM functions that replace the async OpenAI ones
def hf_qwen3_complete(prompt: str, system_prompt: Optional[str] = None, 
                      history_messages: List[Dict] = [], **kwargs) -> str:
    """Synchronous completion function using Hugging Face Qwen3"""
    llm = get_hf_llm_instance()
    result = llm.generate(prompt, system_prompt=system_prompt, **kwargs)
    return result

def hf_embedding(texts: List[str]) -> np.ndarray:
    """Synchronous embedding function using Hugging Face model"""
    embedding_model = get_hf_embedding_instance()
    return embedding_model.embed(texts)

# Wrap embedding function with required attributes
def wrap_hf_embedding_func(embedding_dim: int, max_token_size: int):
    def decorator(func):
        class EmbeddingFuncWrapper:
            def __init__(self, func, embedding_dim, max_token_size):
                self.func = func
                self.embedding_dim = embedding_dim
                self.max_token_size = max_token_size
            
            def __call__(self, *args, **kwargs):
                return self.func(*args, **kwargs)
        
        return EmbeddingFuncWrapper(func, embedding_dim, max_token_size)
    return decorator

# Create the embedding function with proper attributes
hf_embedding_func = wrap_hf_embedding_func(
    embedding_dim=get_hf_embedding_instance().embedding_dim,
    max_token_size=get_hf_embedding_instance().max_token_size
)(hf_embedding)

@dataclass
class HiRAG:
    working_dir: str = field(
        default_factory=lambda: f"./hirag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    # graph mode
    enable_local: bool = True
    enable_naive_rag: bool = False
    enable_hierachical_mode: bool = True

    # text chunking
    chunk_func: Callable[
        [
            List[List[int]],
            List[str],
            tiktoken.Encoding,
            Optional[int],
            Optional[int],
        ],
        List[Dict[str, Union[str, int]]],
    ] = chunking_by_token_size
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"

    # entity extraction
    entity_extract_max_gleaning: int = 1
    entity_summary_to_max_tokens: int = 500

    # graph clustering
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF

    # node embedding
    node_embedding_algorithm: str = "node2vec"
    node2vec_params: dict = field(
        default_factory=lambda: {
            "dimensions": 1536,
            "num_walks": 10,
            "walk_length": 40,
            "num_walks": 10,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        }
    )

    # community reports
    special_community_report_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # text embedding
    embedding_func: callable = field(default_factory=lambda: hf_embedding_func)
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 1  # Keep it synchronous
    query_better_than_threshold: float = 0.2

    # LLM
    using_azure_openai: bool = False
    best_model_func: callable = hf_qwen3_complete  # Use Hugging Face model
    best_model_max_token_size: int = 32768
    best_model_max_async: int = 1  # Keep it synchronous
    cheap_model_func: callable = hf_qwen3_complete  # Use Hugging Face model
    cheap_model_max_token_size: int = 32768
    cheap_model_max_async: int = 1  # Keep it synchronous

    # entity extraction
    entity_extraction_func: callable = None  # Will be set later
    hierarchical_entity_extraction_func: callable = None  # Will be set later

    # storage
    key_string_value_json_storage_cls: type = JsonKVStorage
    vector_db_storage_cls: type = NanoVectorDBStorage
    vector_db_storage_cls_kwargs: dict = field(default_factory=dict)
    graph_storage_cls: type = NetworkXStorage
    enable_llm_cache: bool = True

    # extension
    always_create_working_dir: bool = True
    addon_params: dict = field(default_factory=dict)
    convert_response_to_json_func: callable = None  # Will be set later

    def __post_init__(self):
        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in self.__dict__.items()])
        logger.debug(f"HiRAG init with param:\n\n  {_print_config}\n")

        if not os.path.exists(self.working_dir) and self.always_create_working_dir:
            logger.info(f"Creating working directory {self.working_dir}")
            os.makedirs(self.working_dir)

        self.full_docs = self.key_string_value_json_storage_cls(
            namespace="full_docs", global_config=self.__dict__
        )

        self.text_chunks = self.key_string_value_json_storage_cls(
            namespace="text_chunks", global_config=self.__dict__
        )

        self.llm_response_cache = (
            self.key_string_value_json_storage_cls(
                namespace="llm_response_cache", global_config=self.__dict__
            )
            if self.enable_llm_cache
            else None
        )

        self.community_reports = self.key_string_value_json_storage_cls(
            namespace="community_reports", global_config=self.__dict__
        )
        self.chunk_entity_relation_graph = self.graph_storage_cls(
            namespace="chunk_entity_relation", global_config=self.__dict__
        )

        # No async limiting since we're using synchronous functions
        self.entities_vdb = (
            self.vector_db_storage_cls(
                namespace="entities",
                global_config=self.__dict__,
                embedding_func=self.embedding_func,
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        self.chunks_vdb = (
            self.vector_db_storage_cls(
                namespace="chunks",
                global_config=self.__dict__,
                embedding_func=self.embedding_func,
            )
            if self.enable_naive_rag
            else None
        )

    def insert(self, string_or_strings):
        """Synchronous insert function"""
        start_time = time.time()
        logger.info("Starting document insertion process...")
        
        # Measure total insertion time
        try:
            result = self._sync_insert(string_or_strings)
            total_time = time.time() - start_time
            logger.info(f"Insertion completed in {total_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Insertion failed: {str(e)}")
            raise

    def _sync_insert(self, string_or_strings):
        """Internal synchronous insert implementation"""
        if isinstance(string_or_strings, str):
            string_or_strings = [string_or_strings]
        
        # ---------- new docs
        logger.info(f"Processing {len(string_or_strings)} documents")
        new_docs = {    # dict: {hash: ori_content}
            compute_mdhash_id(c.strip(), prefix="doc-"): {"content": c.strip()}
            for c in string_or_strings
        }
        
        _add_doc_keys = self.full_docs.filter_keys(list(new_docs.keys()))     # filter the docs that has already in the storage.
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        if not len(new_docs):
            logger.warning(f"All docs are already in the storage")
            return
        
        logger.info(f"[New Docs] inserting {len(new_docs)} docs")
        
        # ---------- chunking
        logger.info("Starting chunking process...")
        chunk_start_time = time.time()
        
        inserting_chunks = get_chunks(
            new_docs=new_docs,
            chunk_func=self.chunk_func,
            overlap_token_size=self.chunk_overlap_token_size,
            max_token_size=self.chunk_token_size,
        )
        
        chunk_time = time.time() - chunk_start_time
        logger.info(f"Chunking completed in {chunk_time:.2f} seconds")
        
        _add_chunk_keys = self.text_chunks.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            logger.warning(f"All chunks are already in the storage")
            return
        logger.info(f"[New Chunks] inserting {len(inserting_chunks)} chunks")
        if self.enable_naive_rag:
            logger.info("Insert chunks for naive RAG")
            self.chunks_vdb.upsert(inserting_chunks)

        # TODO: no incremental update for communities now, so just drop all
        self.community_reports.drop()                             # empty the data

        # ---------- extract/summary entity and upsert to graph
        if not self.enable_hierachical_mode:
            logger.info("\033[94m[[Entity Extraction]...\033[0m")
            entity_start_time = time.time()
            maybe_new_kg = self.entity_extraction_func(
                inserting_chunks,
                knwoledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=self.__dict__,
            )
            entity_time = time.time() - entity_start_time
            logger.info(f"Entity extraction completed in {entity_time:.2f} seconds")
        else:
            logger.info("\033[94m[Hierachical Entity Extraction]...\033[0m")
            entity_start_time = time.time()
            maybe_new_kg = self.hierarchical_entity_extraction_func(
                inserting_chunks,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                global_config=self.__dict__,
            )
            entity_time = time.time() - entity_start_time
            logger.info(f"Hierarchical entity extraction completed in {entity_time:.2f} seconds")
            
        if maybe_new_kg is None:
            logger.warning("No new entities found")
            return
        self.chunk_entity_relation_graph = maybe_new_kg
        
        # ---------- update clusterings of graph
        logger.info("\033[94m[Community Report]...\033[0m")
        clustering_start_time = time.time()
        self.chunk_entity_relation_graph.clustering(
            self.graph_cluster_algorithm                    # use leiden
        )
        clustering_time = time.time() - clustering_start_time
        logger.info(f"Graph clustering completed in {clustering_time:.2f} seconds")
        
        community_start_time = time.time()
        generate_community_report(
            self.community_reports, self.chunk_entity_relation_graph, self.__dict__
        )
        community_time = time.time() - community_start_time
        logger.info(f"Community report generation completed in {community_time:.2f} seconds")

        # ---------- commit upsertings and indexing
        logger.info("Committing changes to storage...")
        commit_start_time = time.time()
        self.full_docs.upsert(new_docs)
        self.text_chunks.upsert(inserting_chunks)
        commit_time = time.time() - commit_start_time
        logger.info(f"Commit completed in {commit_time:.2f} seconds")

    def query(self, query: str, param: QueryParam = QueryParam()):
        """Synchronous query function"""
        start_time = time.time()
        logger.info(f"Starting query: {query[:50]}...")
        
        try:
            result = self._sync_query(query, param)
            total_time = time.time() - start_time
            logger.info(f"Query completed in {total_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def _sync_query(self, query: str, param: QueryParam = QueryParam()):
        """Internal synchronous query implementation"""
        if param.mode == "naive" and not self.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "hi" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical mode")
        if param.mode == "hi_nobridge" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_nobridge mode")
        if param.mode == "hi_bridge" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_bridge mode")
        if param.mode == "hi_local" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_local mode")
        if param.mode == "hi_global" and not self.enable_hierachical_mode:
            raise ValueError("enable_hierachical_mode is False, cannot query in hierarchical_global mode")

        # Import query functions here to avoid circular imports
        from hirag._op import (
            hierarchical_query,
            hierarchical_bridge_query,
            hierarchical_local_query,
            hierarchical_global_query,
            hierarchical_nobridge_query,
            naive_query
        )

        if param.mode == "hi":                        # retrieve with hierarchical knowledge
            response = hierarchical_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.__dict__,
            )
        elif param.mode == "hi_bridge":                 # retrieve with only bridge knowledge
            response = hierarchical_bridge_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.__dict__,
            )
        elif param.mode == "hi_local":                  # retrieve with only local knowledge
            response = hierarchical_local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.__dict__,
            )
        elif param.mode == "hi_global":                 # retrieve with only global knowledge
            response = hierarchical_global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.__dict__,
            )
        elif param.mode == "hi_nobridge":               # retrieve with no bridge knowledge
            response = hierarchical_nobridge_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param,
                self.__dict__,
            )
        elif param.mode == "naive":                     # retrieve with only text units
            response = naive_query(
                query,
                self.chunks_vdb,
                self.text_chunks,
                param,
                self.__dict__,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        
        self._query_done()
        return response

    def _query_done(self):
        """Finalize query operations"""
        if self.llm_response_cache:
            self.llm_response_cache.index_done_callback()

def load_cs_data(file_path: str):
    """Load the CS dataset from JSON file"""
    logger.info(f"Loading CS dataset from {file_path}")
    start_time = time.time()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    load_time = time.time() - start_time
    logger.info(f"Dataset loaded in {load_time:.2f} seconds. Total documents: {len(data)}")
    
    return data

def main():
    """Main execution function for the Kaggle notebook system"""
    logger.info("Initializing HiRAG system with Hugging Face Qwen3 8B model...")
    
    # Load the CS dataset
    cs_data_path = "/workspace/eval/datasets/cs/cs_unique_contexts.json"
    documents = load_cs_data(cs_data_path)
    
    # Initialize HiRAG with Hugging Face models
    start_time = time.time()
    hirag_system = HiRAG(
        working_dir=f"./kaggle_hirag_qwen3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        best_model_func=hf_qwen3_complete,
        cheap_model_func=hf_qwen3_complete,
        embedding_func=hf_embedding_func,
        enable_hierachical_mode=True,
        enable_local=True,
        enable_naive_rag=False,
        chunk_token_size=1000,  # Adjust based on model's context window
        chunk_overlap_token_size=100
    )
    
    # Set the required functions that were None initially
    hirag_system.hierarchical_entity_extraction_func = extract_hierarchical_entities
    hirag_system.entity_extraction_func = extract_entities
    hirag_system.convert_response_to_json_func = lambda x: x  # Simple identity function for now
    
    init_time = time.time() - start_time
    logger.info(f"HiRAG system initialized in {init_time:.2f} seconds")
    
    # Insert documents into the system
    logger.info("Starting document insertion into HiRAG system...")
    insertion_start_time = time.time()
    
    # Process documents in batches to manage memory
    batch_size = 5  # Small batch size for memory management
    total_docs = len(documents)
    
    for i in tqdm(range(0, total_docs, batch_size), desc="Processing document batches"):
        batch_docs = documents[i:i+batch_size]
        hirag_system.insert(batch_docs)
        
    insertion_time = time.time() - insertion_start_time
    logger.info(f"All documents inserted in {insertion_time:.2f} seconds")
    
    # Example query to test the system
    logger.info("Testing system with example query...")
    test_query = "What are the main topics covered in the computer science documents?"
    query_param = QueryParam(mode="hi")  # Use hierarchical mode
    
    query_start_time = time.time()
    response = hirag_system.query(test_query, query_param)
    query_time = time.time() - query_start_time
    
    logger.info(f"Query completed in {query_time:.2f} seconds")
    logger.info(f"Response: {response}")
    
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    return hirag_system, response

if __name__ == "__main__":
    # Run the system
    system, result = main()
    print("System execution completed successfully!")
    print(f"Example query result: {result}")