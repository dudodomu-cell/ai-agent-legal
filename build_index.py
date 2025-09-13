# build_index.py
import os, json, pickle, math
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from openai import OpenAI

PDF_PATH = Path("mica.pdf")  # твій файл
OUT_DIR = Path("data"); OUT_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 900      # ~900 символів
CHUNK_OVERLAP = 150   # перекриття
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

def chunk_text(text, page):
    chunks, step = [], CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, max(1, len(text)), step):
        part = text[i:i+CHUNK_SIZE]
        if part.strip():
            chunks.append({"text": part, "page": page})
    return chunks

def load_pdf_chunks(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    all_chunks = []
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        t = " ".join(t.split())
        all_chunks += chunk_text(t, page=i)
    return all_chunks

def embed_texts(texts):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # OpenAI повертає np.float32-вектори
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return arr

def main():
    assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"
    print("Loading PDF...")
    chunks = load_pdf_chunks(PDF_PATH)
    print("Chunks:", len(chunks))

    print("Embedding...")
    batch = 96
    vecs_list = []
    for i in range(0, len(chunks), batch):
        vecs = embed_texts([c["text"] for c in chunks[i:i+batch]])
        vecs_list.append(vecs)
    mat = np.vstack(vecs_list)  # (N, d)

    # Збережемо в FAISS-подібному форматі без зовнішніх залежностей (простий пошук косинусом)
    np.save(OUT_DIR / "vectors.npy", mat)
    with open(OUT_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("Done -> data/vectors.npy, data/chunks.pkl")

if __name__ == "__main__":
    main()