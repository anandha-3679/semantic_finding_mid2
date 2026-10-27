import os, re
import numpy as np
from openai import OpenAI

_client = None
def client():
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"])
    return _client

EMBED_MODEL = lambda: os.getenv("EMBED_MODEL", "openai/text-embedding-3-small")
CHAT_MODEL = lambda: os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")

def _windows(text, max_words=120, overlap=30):
    words = text.split()
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, max(len(words), 1), step)]

def chunk(paragraphs, max_chars=700):
    chunks = []
    for page, text in paragraphs:
        text = text.strip()
        if len(text) < 20:
            continue
        if len(text) <= max_chars:
            chunks.append({"page": page, "text": text})
            continue
        # 1) try sentence-based pieces
        pieces, buf = [], ""
        for s in re.split(r"(?<=[.!?:;])\s+|\s[•●▪\-–]\s", text):
            if buf and len(buf) + len(s) > max_chars:
                pieces.append(buf); buf = ""
            buf += (" " if buf else "") + s
        if buf:
            pieces.append(buf)
        # 2) any piece still too long -> fixed word windows
        for p in pieces:
            if len(p) > max_chars * 1.5:
                for w in _windows(p):
                    chunks.append({"page": page, "text": w})
            else:
                chunks.append({"page": page, "text": p})
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks

def embed(texts):
    vecs = []
    for i in range(0, len(texts), 64):
        r = client().embeddings.create(model=EMBED_MODEL(), input=texts[i:i + 64])
        vecs.extend(d.embedding for d in sorted(r.data, key=lambda d: d.index))
    m = np.array(vecs, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)  # normalise -> dot = cosine

def search(matrix, query, top_k=5):
    q = embed([query])[0]
    scores = matrix @ q
    idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in idx]

def explain(query, passage):
    try:
        r = client().chat.completions.create(
            model=CHAT_MODEL(),
            messages=[
                {"role": "system", "content": "In one short sentence, explain how the passage relates to the user's text. Use only the passage."},
                {"role": "user", "content": f"User text: {query}\n\nPassage: {passage}"},
            ],
            max_tokens=80,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return ""