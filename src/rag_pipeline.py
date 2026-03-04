"""RAG pipeline with LangChain for question answering."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import Config
from .vector_store import SyllabusVectorStore


class RAGPipeline:

    PROMPT_TEMPLATE = """
You are an expert academic assistant specializing in university syllabi.

Use ONLY the following context from uploaded syllabus documents.

If the answer cannot be found in the context say:
"I don't have enough information in the uploaded syllabus documents."

Context:
{context}

Question:
{question}

Answer:
"""

    def __init__(self, vector_store: SyllabusVectorStore):

        Config.validate()

        self.vector_store = vector_store

        self.retriever = self.vector_store.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": Config.TOP_K_RESULTS,
                "fetch_k": Config.TOP_K_RESULTS * 2,
                "lambda_mult": 0.5
            }
        )

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.1,
            openai_api_key=Config.OPENROUTER_API_KEY,
            base_url=Config.OPENROUTER_BASE_URL
        )

        self.prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)

    def answer_question(self, question: str) -> Dict[str, Any]:

        docs: List[Document] = self.retriever.invoke(question)

        if not docs:
            return {
                "answer": "I don't have enough information in the uploaded syllabus documents.",
                "sources": []
            }

        context = "\n\n".join([doc.page_content for doc in docs])

        messages = self.prompt.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(messages)

        return {
            "answer": response.content,
            "sources": [doc.metadata.get("source") for doc in docs]
        }