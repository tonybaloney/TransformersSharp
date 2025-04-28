from typing import Any, Optional
from transformers import pipeline as TransformersPipeline, Pipeline, TextGenerationPipeline
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from collections.abc import Buffer

def pipeline(task: Optional[str] = None, model: Optional[str] = None, tokenizer: Optional[str] = None, torch_dtype: Optional[str] = None, device: Optional[str] = None):
    if torch_dtype is not None:
        if not hasattr(torch, torch_dtype.lower()):
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        else:
            torch_dtype = getattr(torch, torch_dtype.lower())
    return TransformersPipeline(task=task, model=model, tokenizer=tokenizer, torch_dtype=torch_dtype, device=device)


def text_generation_pipeline(pipeline: TextGenerationPipeline, 
                             messages: list[dict[str, str]],
                             max_length: Optional[int] = None,
                             max_new_tokens: Optional[int] = None,
                             min_length: Optional[int] = None,
                             min_new_tokens: Optional[int] = None,
                             stop_strings: Optional[list[str]] = None,
                             temperature: Optional[float] = 1.0,
                             top_k: Optional[int] = 50,
                             top_p: Optional[float] = 1.0,
                             min_p: Optional[float] = None,
                            ) -> list[dict[str, Any]]:
    return pipeline(messages, max_length=max_length, max_new_tokens=max_new_tokens, min_length=min_length, min_new_tokens=min_new_tokens, stop=stop_strings, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)

def huggingface_login(token: str) -> None:
    login(token=token)


def call_pipeline(pipeline: Pipeline, input: str, **kwargs) -> list[dict[str, Any]]:
    return pipeline(input, **kwargs)


def call_pipeline_with_list(pipeline: Pipeline, input: list[str], **kwargs) -> list[dict[str, Any]]:
    return pipeline(input, **kwargs)


def tokenizer_from_pretrained(model: str, 
                              cache_dir: Optional[str] = None, 
                              force_download: bool = False, 
                              revision: Optional[str] = 'main', 
                              trust_remote_code: bool  = False) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, force_download=force_download, revision=revision, trust_remote_code=trust_remote_code)


def tokenize_text_with_attention(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[list[int], list[int]]:
    result = tokenizer(text)
    return result['input_ids'], result['attention_mask']

def tokenizer_text_as_ndarray(tokenizer: PreTrainedTokenizerBase, text: str, **kwargs) -> Buffer:
    result = tokenizer(text, return_tensors='np', return_attention_mask=False)
    return result['input_ids'][0]
