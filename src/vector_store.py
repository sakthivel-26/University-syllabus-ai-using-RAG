"""ChromaDB vector store."""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma

from .config import Config
from .embeddings import get_embeddings


class SyllabusVectorStore:

    def __init__(self):

        self.embeddings = get_embeddings()

        self.vector_store = Chroma(
            collection_name=Config.COLLECTION_NAME,
            persist_directory=Config.CHROMA_DB_PATH,
            embedding_function=self.embeddings
        )

    # -----------------------------
    # Add Documents
    # -----------------------------

    def add_documents(self, docs: List[Document]):

        return self.vector_store.add_documents(docs)

    # -----------------------------
    # Duplicate Document Check
    # -----------------------------

    def document_exists(self, document_hash: str) -> bool:
        """
        Check if a document with this hash already exists in the vector store.
        """

        try:

            results = self.vector_store._collection.get(
                where={"document_hash": document_hash},
                limit=1
            )

            if results and results.get("ids"):
                return True

            return False

        except Exception:
            return False

    # -----------------------------
    # Similarity Search
    # -----------------------------

    def similarity_search(self, query: str, k: Optional[int] = None):

        k = k or Config.TOP_K_RESULTS

        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: Optional[int] = None):

        k = k or Config.TOP_K_RESULTS

        return self.vector_store.similarity_search_with_score(query, k=k)

    # -----------------------------
    # Collection Info
    # -----------------------------

    def get_collection_info(self) -> Dict[str, Any]:

        try:

            count = self.vector_store._collection.count()

            return {
                "collection_name": Config.COLLECTION_NAME,
                "document_count": count
            }

        except Exception:

            return {
                "collection_name": Config.COLLECTION_NAME,
                "document_count": 0
            }

    # -----------------------------
    # Delete Collection
    # -----------------------------

    def delete_collection(self):

        try:

            self.vector_store._client.delete_collection(Config.COLLECTION_NAME)

            # recreate empty collection
            self.vector_store = Chroma(
                collection_name=Config.COLLECTION_NAME,
                persist_directory=Config.CHROMA_DB_PATH,
                embedding_function=self.embeddings
            )

        except Exception as e:

            print(f"Error deleting collection: {e}")