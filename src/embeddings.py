"""Embedding generation using SentenceTransformers."""

from langchain_huggingface import HuggingFaceEmbeddings
from .config import Config


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME
    )