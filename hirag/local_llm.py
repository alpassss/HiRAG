import asyncio
import logging
from typing import List, Dict, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError
import json

logger = logging.getLogger("HiRAG")

LLM_MODEL_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    # Test if torch and transformers can actually be used
    if torch.cuda.is_available():
        logger.info("CUDA is available for torch")
    LLM_MODEL_AVAILABLE = True
    logger.info("Successfully imported transformers and torch")
except Exception as e:
    logger.warning(f"transformers or torch not available ({e}), using mock LLM")
    LLM_MODEL_AVAILABLE = False

class LocalLLM:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.generator = None
        self.tokenizer = None
        
        if LLM_MODEL_AVAILABLE:
            try:
                # Try to load a lightweight model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.generator = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add padding token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Failed to load LLM model {model_name}: {e}")
                self.generator = None
                self.tokenizer = None

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using local model or mock response
        """
        if self.generator is not None and self.tokenizer is not None:
            try:
                # Combine system prompt and user prompt if system prompt exists
                full_prompt = prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                
                inputs = self.tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.generator.generate(
                        inputs, 
                        max_length=min(inputs.shape[1] + 150, 1024),
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the generated part (after the original prompt)
                if response.startswith(full_prompt):
                    response = response[len(full_prompt):].strip()
                else:
                    # If the model didn't generate after the prompt, return the last part
                    response = response.split(full_prompt)[-1].strip()
                
                return response if response else "I understand your query, but I cannot provide a detailed response."
            except Exception as e:
                logger.warning(f"Local model generation failed: {e}, using mock response")
        
        # Fallback: return a mock response based on the prompt
        logger.warning("Using mock LLM response as local model is not available")
        return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """
        Generate a mock response based on the prompt
        """
        # Simple mock responses based on prompt content
        prompt_lower = prompt.lower()
        
        if "entity" in prompt_lower or "extract" in prompt_lower:
            return '"entity", "PERSON", "A person mentioned in the text", 0.9'
        elif "relationship" in prompt_lower or "relation" in prompt_lower:
            return '"relationship", "PERSON", "PERSON", "They are connected", 0.8'
        elif "summarize" in prompt_lower or "summary" in prompt_lower:
            return "This is a summary of the content provided."
        else:
            return "This is a generated response based on the prompt provided. The local model is not available, so a mock response is returned."

# Global instance
local_llm_instance = None

def get_local_llm():
    global local_llm_instance
    if local_llm_instance is None:
        local_llm_instance = LocalLLM()
    return local_llm_instance

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def local_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """
    Local completion function that can be used instead of OpenAI's API
    """
    local_llm = get_local_llm()
    return await local_llm.generate(prompt, system_prompt=system_prompt, **kwargs)

# Specific model functions
async def local_gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await local_complete_if_cache(
        "local-model",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def local_gpt_35_turbo_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await local_complete_if_cache(
        "local-model",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def local_gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await local_complete_if_cache(
        "local-model",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )