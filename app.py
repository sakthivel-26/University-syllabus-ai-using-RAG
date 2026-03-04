import os
import asyncio
import json
import hashlib
import shutil
from typing import List, Tuple

import gradio as gr
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF

# ---------------- Azure OpenAI Config ----------------
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = "./cache"

SYSTEM_PROMPT = """
You are a university-level academic assistant.
Explain answers clearly with structured reasoning.
Use simple but formal academic language.
Provide accurate, well-structured explanations.
"""

os.makedirs(CACHE_DIR, exist_ok=True)

embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

DOCS: List[str] = []
FILENAMES: List[str] = []
EMBEDDINGS: np.ndarray = None
FAISS_INDEX = None


# ---------------- PDF Extraction ----------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


# ---------------- FAISS ----------------
def build_faiss(emb: np.ndarray):
    global FAISS_INDEX
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype("float32"))
    FAISS_INDEX = index


def search(query: str, k: int = 3):
    if FAISS_INDEX is None:
        return []

    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = FAISS_INDEX.search(q_emb, k)

    results = []
    for i in I[0]:
        if i >= 0:
            results.append({
                "text": DOCS[i][:15000],
                "source": FILENAMES[i]
            })
    return results


# ---------------- Azure OpenAI Call ----------------
def call_azure_openai(prompt: str):

    if not AZURE_OPENAI_API_KEY:
        return "Azure API key missing."

    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }

    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        return f"Azure error: {response.text}"

    data = response.json()
    return data["choices"][0]["message"]["content"]


# ---------------- Upload & Index ----------------
def upload_and_index(files):
    global DOCS, FILENAMES, EMBEDDINGS

    if not files:
        return "No files uploaded."

    DOCS = []
    FILENAMES = []

    for f in files:
        with open(f.name, "rb") as file:
            content = file.read()
        DOCS.append(extract_text_from_pdf(content))
        FILENAMES.append(f.name)

    EMBEDDINGS = embedder.encode(DOCS, convert_to_numpy=True)
    build_faiss(EMBEDDINGS)

    return f"Indexed {len(DOCS)} PDFs successfully."


# ---------------- Ask Question ----------------
def ask(question):
    if not DOCS:
        return "Upload PDFs first."

    results = search(question)

    context = "\n\n".join(
        f"Source: {r['source']}\n{r['text']}" for r in results
    )

    prompt = f"""
Use the academic context below to answer clearly.

Context:
{context}

Question:
{question}

Provide a well-structured university-level explanation.
"""

    return call_azure_openai(prompt)


# ---------------- Gradio UI ----------------
with gr.Blocks(title="Azure PDF RAG Bot") as demo:
    gr.Markdown("# 🎓 University-Level PDF RAG Bot (Azure OpenAI GPT-4.1)")

    file_input = gr.File(file_count="multiple", file_types=[".pdf"])
    upload_btn = gr.Button("Upload & Index")
    status = gr.Textbox(label="Status")

    upload_btn.click(upload_and_index, inputs=file_input, outputs=status)

    gr.Markdown("## Ask a Question")
    question = gr.Textbox(lines=3)
    ask_btn = gr.Button("Ask")
    answer = gr.Textbox(lines=15)

    ask_btn.click(ask, inputs=question, outputs=answer)

if __name__ == "__main__":
    demo.launch()