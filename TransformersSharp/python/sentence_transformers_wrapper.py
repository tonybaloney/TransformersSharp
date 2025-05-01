from sentence_transformers import SentenceTransformer
from collections.abc import Buffer
from typing import Optional


def sentence_transformer(model: str, 
                         device: Optional[str] = None,
                         cache_dir: Optional[str] = None, 
                         revision: Optional[str] = 'main', 
                         trust_remote_code: bool = False) -> SentenceTransformer:
    """
    Load a SentenceTransformer model.
    """
    return SentenceTransformer(model, device=device, cache_folder=cache_dir, revision=revision, trust_remote_code=trust_remote_code)


def encode_sentence(model: SentenceTransformer, sentence: str) -> Buffer:
    """
    Encode a list of sentences using the SentenceTransformer model.
    """
    return model.encode([sentence])[0]


def encode_sentences(model: SentenceTransformer, sentences: list[str]) -> Buffer:
    """
    Encode a list of sentences using the SentenceTransformer model.
    """
    return model.encode(sentences)
