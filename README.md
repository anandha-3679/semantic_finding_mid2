# Semantic Finder

Find the passage in a document that matches a sentence or an *idea*, even when no keywords overlap. Upload a PDF, DOCX or TXT file, type a sentence, and the app highlights the most relevant passages in the document.

It works like a keyword search where you only know the concept. The document is split into passages, each passage and your query become embeddings (vectors of meaning), and the closest passages are returned. An LLM adds a one-line explanation of why the top result matches (RAG).

## Features

- Semantic (meaning-based) search, not exact keyword match
- PDF, DOCX, TXT and MD support
- OCR for scanned or image-only PDFs
- Matches highlighted in a full document view, with scores and page numbers
- Models served through OpenRouter

## Tech stack

- Backend: FastAPI, NumPy
- Embeddings and LLM: OpenRouter (OpenAI-compatible API)
- Parsing: PyMuPDF, python-docx, RapidOCR
- Frontend: plain HTML and JavaScript

## Project structure

```
semantic_finder/
├── backend/
│   ├── app/
│   │   ├── main.py      API routes, serves the UI
│   │   ├── parser.py    PDF / DOCX / TXT to paragraphs (with OCR fallback)
│   │   └── rag.py       chunking, embeddings, search, explanation
│   ├── requirements.txt
│   └── .env             your API key (not committed)
├── frontend/
│   └── index.html
├── .gitignore
└── README.md
```

## Setup

Requires Python 3.10+ and an [OpenRouter](https://openrouter.ai) API key.

```bash
git clone https://github.com/anandha-3679/semantic_finding_mid2.git
cd
