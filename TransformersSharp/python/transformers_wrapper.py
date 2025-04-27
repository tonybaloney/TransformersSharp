from typing import Any, Optional
from transformers import pipeline as TransformersPipeline, Pipeline
from huggingface_hub import login
import torch


def pipeline(task: Optional[str] = None, model: Optional[str] = None, tokenizer: Optional[str] = None, torch_dtype: Optional[str] = None, device: Optional[str] = None):
    if torch_dtype is not None:
        if not hasattr(torch, torch_dtype.lower()):
            raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")
        else:
            torch_dtype = getattr(torch, torch_dtype.lower())
    return TransformersPipeline(task=task, model=model, tokenizer=tokenizer, torch_dtype=torch_dtype, device=device)


def huggingface_login(token: str) -> None:
    login(token=token)


def call_pipeline(pipeline: Pipeline, input: str, **kwargs) -> list[dict[str, Any]]:
    return pipeline(input, **kwargs)


def call_pipeline_with_list(pipeline: Pipeline, input: list[str], **kwargs) -> list[list[dict[str, Any]]]:
    return pipeline(input, **kwargs)

