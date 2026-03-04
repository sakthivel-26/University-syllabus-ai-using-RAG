"""PDF document processing and chunking."""

import hashlib
from pathlib import Path
from typing import List
import pypdf

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import Config


class DocumentProcessor:

    def __init__(self):

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:

        text = []

        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)

            if not reader.pages:
                raise ValueError("PDF empty")

            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text.append(content)

        return "\n".join(text)

    def process_pdf(self, pdf_path: Path) -> List[Document]:

        text = self.extract_text_from_pdf(pdf_path)

        doc_hash = hashlib.sha256(text.encode()).hexdigest()

        chunks = self.text_splitter.split_text(text)

        documents = []

        for i, chunk in enumerate(chunks):

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path.name,
                        "chunk_index": i,
                        "document_hash": doc_hash
                    },
                )
            )

        return documents


    # ✅ ADD THIS METHOD
    def get_document_hash(self, pdf_path: Path) -> str:
        """
        Generate SHA256 hash for a PDF document.
        Used to check if document already exists.
        """

        text = self.extract_text_from_pdf(pdf_path)

        return hashlib.sha256(text.encode()).hexdigest()