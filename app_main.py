from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pypdf import PdfReader
from fastapi import Query

app = FastAPI()
client = OpenAI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"ok": True}

# ---------- ✅ ПРОСТИЙ ІНДЕКС PDF (КАШ У ПАМ’ЯТІ) ----------
PDF_PATH = "mica.pdf"  # якщо інша назва — підправ
_PAGES: list[dict] = []

def _load_pdf_once():
    global _PAGES
    if _PAGES:
        return
    try:
        reader = PdfReader(PDF_PATH)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # трішки чистки: прибираємо зайві пропуски
            text = " ".join(text.split())
            pages.append({"page": i + 1, "text": text})
        _PAGES = pages
        print(f"PDF loaded: {len(_PAGES)} pages")
    except Exception as e:
        print("PDF LOAD ERROR:", e)
        _PAGES = []

def _score_page(query: str, page_text: str) -> int:
    # Дуже простий скоринг: рахуємо збіги слів (укр/рос/англ норм)
    q_words = [w for w in query.lower().split() if len(w) > 2]
    txt = page_text.lower()
    return sum(txt.count(w) for w in q_words)

def _find_snippets(query: str, k: int = 3, window: int = 400):
    """Шукає слова запиту і вирізає уривки довкола збігів."""
    _load_pdf_once()
    if not _PAGES:
        return []

    q_words = [w for w in query.lower().split() if len(w) > 2]
    hits = []

    for p in _PAGES:
        text = p["text"].lower()
        for w in q_words:
            idx = text.find(w)
            if idx != -1:
                start = max(0, idx - window // 2)
                end = min(len(p["text"]), idx + window // 2)
                snippet = p["text"][start:end]
                hits.append({"page": p["page"], "text": snippet})

    # залишаємо лише кілька найрелевантніших
    return hits[:k]

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        snippets = _find_snippets(req.question, k=3)
        if not snippets:
            system = ("Ти — юридичний асистент. Якщо в наданому контексті немає відповіді — чесно скажи, що не знайшов.")
            user = f"Питання: {req.question}\n\nКонтекст: (немає релевантних уривків)"
        else:
            ctx = "\n\n".join([f"[page {s['page']}]\n{s['text']}" for s in snippets])
            system = ("Ти — юридичний асистент. Відповідай ТІЛЬКИ на основі наведеного контексту з MiCA. "
                      "Якщо відповіді немає у контексті — скажи, що не знайшов. "
                      "Наприкінці наведи список сторінок у форматі [page N].")
            user = f"Питання: {req.question}\n\nКонтекст із MiCA:\n{ctx}"

        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        answer = r.choices[0].message.content.strip()
        disclaimer = "⚠️ Це не є юридичною консультацією. Для остаточного вирішення зверніться до кваліфікованого юриста."
        return {"answer": f"{answer}\n\n{disclaimer}"}
    except Exception as e:
        return {"error": str(e)}

        from fastapi import Query

@app.get("/chat_get")
def chat_get(q: str = Query(..., description="question")):
    # Використовуємо ту ж функцію, що і для POST
    return chat(ChatRequest(question=q))


if __name__ == "__main__":
    import uvicorn
    # (на час розробки тримаємо лог у консолі)
    uvicorn.run(app, host="127.0.0.1", port=8000)