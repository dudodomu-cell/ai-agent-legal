from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
import os
from dotenv import load_dotenv
from typing import Optional
from fastapi import Query

# load .env once on startup

load_dotenv()

def env_list(name: str, default_list):
    v = os.getenv(name)
    if not v:
        return list(default_list)
    return [x.strip() for x in v.split(",") if x.strip()]

def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1","true","yes","on")

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except:
        return default

# читаємо змінні з .env
APP_VERSION    = os.getenv("APP_VERSION", "0.1.1")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # старий параметр
DEFAULT_MODEL  = os.getenv("DEFAULT_MODEL", OPENAI_MODEL)   # новий дефолт
ALLOWED_MODELS = env_list("ALLOWED_MODELS", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini","gpt-4.1"])
REQUIRE_CONTEXT = env_bool("REQUIRE_CONTEXT", True)
TOP_K           = env_int("TOP_K", 5)
MIN_SIM         = env_float("MIN_SIM", 0.22)
FAST_DEV        = env_bool("FAST_DEV", True) 
FREE_LIMIT      = env_int("FREE_LIMIT", 5)
          # поріг схожості, щоб вважати збіг реальним

from datetime import datetime

STATS = {
    "total_requests": 0,
    "start_time": datetime.utcnow().isoformat()
}

app = FastAPI()

# CORS: дозволяємо локальні origin-и 5500 (статична сторінка) і 8000 (бекенд)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],   # явний список
    allow_headers=["*"],
)

def gpt_answer(question: str, model_override: str | None = None) -> str:
    ...
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return f"Echo: {question}"

    # вибір моделі з валідацією
    chosen = (model_override or "").strip() if model_override else ""
    if chosen and chosen in ALLOWED_MODELS:
        model = chosen
    else:
        model = DEFAULT_MODEL


    # (як і було)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=12.0)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"You are a legal assistant AI. Answer strictly from the provided MiCA context if available."},
                      {"role":"user","content":prompt if ('prompt' in locals()) else question}],
            temperature=0.0
        )
        ans = resp.choices[0].message.content if resp and resp.choices else ""
        return ans or "Empty answer."
    except Exception as e:
        return f"Echo (OpenAI error: {type(e).name}): {question}"

def gpt_answer(question: str, model_override: str | None = None) -> str:
    question = (question or "").strip()
    if not question:
        return "Empty question."

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return f"Echo: {question}"

    # Модель: override з валідацією
    chosen = (model_override or "").strip()
    model = chosen if (chosen and chosen in ALLOWED_MODELS) else DEFAULT_MODEL

    # ---- RAG ----
    ctx_items = rag_search(question, k=TOP_K) if (VEC is not None and CHUNKS is not None) else []
    ctx_items = [c for c in ctx_items if c[2] >= MIN_SIM]

    if REQUIRE_CONTEXT and not ctx_items:
        return "I cannot answer this from the provided documents (MiCA). Please refine the question."

    system = (
        "You are a legal assistant AI. Answer strictly based only on the provided legal documents (MiCA Regulation). "
        "If the information is not in the context, say you cannot answer."
    )
    prompt = rag_prompt(question, ctx_items) if ctx_items else question

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=12.0)
        msgs = [
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.0)
        ans = resp.choices[0].message.content if resp and resp.choices else ""
        return ans or "Empty answer."
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
    model_req = (req.get("model") or "").strip()
    if not question:
        return JSONResponse({"error": "empty question"}, status_code=400)

    STATS["total_requests"] += 1
    free_left = max(0, FREE_LIMIT - STATS["total_requests"])
    if STATS["total_requests"] > FREE_LIMIT:
        return {
            "answer": "You have reached the free request limit. Please subscribe to continue.",
            "sources": [],
            "usage": {"total_requests": STATS["total_requests"], "free_left": 0, "limit": FREE_LIMIT},
            "model_used": None
        }

    ctx_items = rag_search(question, k=TOP_K)
    ctx_items = [c for c in ctx_items if c[2] >= MIN_SIM]

    # побудова sources (з цитатами)
    if ctx_items:
        src = [{"page": p, "score": round(s,3), "text": (t[:320] + ("…" if len(t) > 320 else ""))} for (t,p,s) in ctx_items]
    else:
        src = []

    # вибір моделі і відповідь
    # (прокидаємо override у gpt_answer)
    answer = gpt_answer(question, model_override=model_req)

    # визначимо фактично використану модель (валідуємо)
    used = model_req if (model_req and model_req in ALLOWED_MODELS) else DEFAULT_MODEL

    return {
        "answer": answer,
        "sources": src,
        "usage": {"total_requests": STATS["total_requests"], "free_left": free_left, "limit": FREE_LIMIT},
        "model_used": used
    }
        

@app.get("/chat_get")
def chat_get(
    q: str = Query(..., description="question"),
    model: Optional[str] = Query(None, description="model name")
):
    # делегуємо до /chat-логіки, але в GET-форматі
    return chat({"question": q, "model": model or ""})

from fastapi.responses import JSONResponse

@app.get("/health")
def health():
    ok = True
    info = {
        "version": APP_VERSION,
        "has_index": bool(VEC is not None and CHUNKS is not None),
        "vec_shape": tuple(VEC.shape) if VEC is not None else None,
        "chunks": len(CHUNKS) if CHUNKS is not None else 0,
    }
    return {"ok": ok, **info}

@app.get("/version")
def version():
    return {"version": APP_VERSION}

from fastapi.responses import JSONResponse


@app.get("/stats")
def stats():
    return STATS

@app.get("/config")
def config():
    return {
        "version": APP_VERSION,
        "require_context": REQUIRE_CONTEXT,
        "top_k": TOP_K,
        "min_sim": MIN_SIM,
        "model_default": DEFAULT_MODEL,
        "models_allowed": ALLOWED_MODELS,
        "has_index": bool(VEC is not None and CHUNKS is not None),
    }

@app.get("/models")
def models():
    return {
        "default": DEFAULT_MODEL,
        "allowed": ALLOWED_MODELS
    }