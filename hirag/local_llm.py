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
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        self.generator = None
        self.tokenizer = None
        
        if LLM_MODEL_AVAILABLE:
            try:
                # Load Qwen3 model with proper settings for H100
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.generator = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                
                # Add padding token if it doesn't exist
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(f"Failed to load LLM model {model_name}: {e}")
                self.generator = None
                self.tokenizer = None

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, history_messages: List[Dict] = None, **kwargs) -> str:
        """
        Generate text using local model or mock response
        """
        if self.generator is not None and self.tokenizer is not None:
            try:
                # Prepare messages for Qwen3 chat format
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                # Add history messages if provided
                if history_messages:
                    messages.extend(history_messages)
                
                # Add current user prompt
                messages.append({"role": "user", "content": prompt})
                
                # Use Qwen3's chat template with thinking enabled
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True  # Enable Qwen3's thinking mode
                )
                
                # Tokenize the input
                model_inputs = self.tokenizer([text], return_tensors="pt")
                
                # Move to model device
                model_inputs = {k: v.to(self.generator.device) for k, v in model_inputs.items()}
                
                # Generate response with appropriate parameters for entity extraction
                max_new_tokens = kwargs.get("max_tokens", 1024)
                temperature = kwargs.get("temperature", 0.7)
                top_p = kwargs.get("top_p", 0.9)
                
                generated_ids = self.generator.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode only the newly generated tokens
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                
                # Parse thinking content if present (Qwen3 uses 151668 as the thinking separator token)
                try:
                    # rindex finding 151668 (thinking separator token)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                    thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                    content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                except ValueError:
                    # If no thinking content found, just decode normally
                    content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                
                # Return the actual content (not the thinking part)
                return content if content else "I understand your query, but I cannot provide a detailed response."
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