import numpy as np
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage

# Import local models
from .local_embedding import local_embedding
from .local_llm import (
    local_gpt_4o_complete,
    local_gpt_35_turbo_complete,
    local_gpt_4o_mini_complete,
    local_complete_if_cache
)

logger = logging.getLogger("HiRAG")

# We don't need OpenAI clients anymore since we're using local models


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),  # Changed from OpenAI-specific exceptions to general Exception
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Use local model instead of OpenAI
    from .local_llm import local_complete_if_cache
    return await local_complete_if_cache(model, prompt, system_prompt, history_messages, **kwargs)


async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_35_turbo_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-3.5-turbo",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
# Use local embedding instead of OpenAI - need to adjust dimensions to match expected 1536
async def openai_embedding(texts: list[str]) -> np.ndarray:
    # Use local embedding and adjust dimensions if needed
    from .local_embedding import local_embedding
    result = await local_embedding(texts)
    
    # If the local embedding has different dimensions (384), we need to pad or transform
    if result.shape[1] != 1536:
        # Create a larger array and fill with repeated values or zeros
        new_result = np.zeros((result.shape[0], 1536), dtype=result.dtype)
        for i in range(result.shape[0]):
            # Repeat the embedding values to fill 1536 dimensions
            for j in range(1536):
                new_result[i, j] = result[i, j % result.shape[1]]
        return new_result
    return result


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),  # Changed from OpenAI-specific exceptions to general Exception
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Use local model instead of Azure OpenAI
    from .local_llm import local_complete_if_cache
    return await local_complete_if_cache(deployment_name, prompt, system_prompt, history_messages, **kwargs)


async def azure_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    from .local_llm import local_gpt_4o_complete
    return await local_gpt_4o_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    from .local_llm import local_gpt_4o_mini_complete
    return await local_gpt_4o_mini_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def azure_openai_embedding(texts: list[str]) -> np.ndarray:
    # Use local embedding instead of Azure OpenAI
    from .local_embedding import local_embedding
    result = await local_embedding(texts)
    
    # If the local embedding has different dimensions (384), we need to pad or transform
    if result.shape[1] != 1536:
        # Create a larger array and fill with repeated values or zeros
        new_result = np.zeros((result.shape[0], 1536), dtype=result.dtype)
        for i in range(result.shape[0]):
            # Repeat the embedding values to fill 1536 dimensions
            for j in range(1536):
                new_result[i, j] = result[i, j % result.shape[1]]
        return new_result
    return result
