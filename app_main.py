from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
import os

app = FastAPI()

# CORS: дозволяємо фронту на 5500 звертатися до бекенду на 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "*"],  # dev-режим
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def gpt_answer(question: str) -> str:
    """
    Повертає відповідь GPT або підміняє echo, якщо:
    - немає OPENAI_API_KEY
    - сталася помилка/таймаут
    """
    question = (question or "").strip()
    if not question:
        return "Порожнє питання."

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # Fallback без ключа
        return f"Echo: {question}"

    try:
        # Новий SDK OpenAI (v1)
        from openai import OpenAI
        # Можеш поміняти модель однією змінною
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Таймаут на весь запит, щоб не висіло
        client = OpenAI(api_key=api_key, timeout=10.0)

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.2,
        )
        if resp and resp.choices:
            return resp.choices[0].message.content or "Відповідь порожня."
        return "Відповідь порожня."
    except Exception as e:
        # Акуратний fallback, щоб UI не висів
        return f"Echo (OpenAI error: {type(e).name}): {question}"

def gpt_answer(question: str) -> str:
    # ... твій існуючий код вище ...
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return f"Echo: {question}"

    # >>>> ДОДАНО: знайдемо релевантний контекст
    ctx = rag_search(question, k=5)
    system = "Ти юридичний помічник. Відповідай лише на основі поданого контексту."
    if ctx:
        prompt = rag_prompt(question, ctx)
    else:
        # Якщо індексу ще нема — відповімо як звичайний чат (тимчасово)
        prompt = question

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=12.0)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # Якщо є контекст — кладемо його в один user-повідомлення (простий варіант)
        messages = [{"role":"system","content":system},
                    {"role":"user","content":prompt}]

        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.0,
        )
        ans = resp.choices[0].message.content if resp and resp.choices else ""
        return ans or "Відповідь порожня."
    except Exception as e:
        return f"Echo (OpenAI error: {type(e).name}): {question}"

# ---- RAG: завантаження індексу ----
from pathlib import Path
import numpy as np, pickle, os
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VEC_PATH = DATA_DIR / "vectors.npy"
CH_PATH  = DATA_DIR / "chunks.pkl"

VEC = None
CHUNKS = None

def _cosine_sim(a, b):
    # a: (d,), b: (N,d)
    na = a / (np.linalg.norm(a) + 1e-9)
    nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return nb @ na

def _embed_query(q: str) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    emb_model = os.getenv("EMB_MODEL", "text-embedding-3-small")
    resp = client.embeddings.create(model=emb_model, input=[q])
    return np.array(resp.data[0].embedding, dtype=np.float32)

def rag_search(query: str, k: int = 5) -> List[Tuple[str,int,float]]:
    """Повертає список (текст_шматка, сторінка, схожість)"""
    if VEC is None or CHUNKS is None:
        return []
    qv = _embed_query(query)
    sims = _cosine_sim(qv, VEC)  # (N,)
    idx = np.argsort(-sims)[:k]
    out = []
    for i in idx:
        c = CHUNKS[i]
        out.append((c["text"], c["page"], float(sims[i])))
    return out

def rag_prompt(query: str, ctx_items: List[Tuple[str,int,float]]) -> str:
    parts = []
    for i,(t,p,s) in enumerate(ctx_items, start=1):
        parts.append(f"[{i}] (MiCA p.{p}, score={s:.2f}) {t}")
    ctx = "\n\n".join(parts) if parts else "NO_CONTEXT"
    return f"""You are a legal assistant restricted to the provided context.
Answer ONLY if the answer is supported by the context. If not found, say you cannot answer from the provided documents.
Cite sources as: (MiCA p.<page>). Keep answers concise and in Ukrainian.

User question: {query}

Context:
{ctx}
"""

# завантажити індекс при старті (якщо є)
try:
    if VEC_PATH.exists() and CH_PATH.exists():
        VEC = np.load(VEC_PATH)
        with open(CH_PATH, "rb") as f:
            CHUNKS = pickle.load(f)
        print(f"RAG index loaded: vecs={VEC.shape}, chunks={len(CHUNKS)}")
    else:
        print("RAG index not found. Run: python build_index.py")
except Exception as e:
    print("RAG load error:", e)        

@app.get("/", response_class=PlainTextResponse)
def root():
    return "ok"

# ====== ОТУТ Є chat() ======
@app.post("/chat")
def chat(req: dict):
    question = (req.get("question") or "").strip()
    if not question:
        return {"error": "empty question"}
    # >>> Ось тут ідемо в GPT (з безпечним fallback усередині)
    answer = gpt_answer(question)
    return {"answer": answer}

@app.get("/chat_get")
def chat_get(q: str = Query(..., description="question")):
    # Використовуємо той самий код, що й /chat, але з GET
    return chat({"question": q})