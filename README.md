# 🧬 MedQuery AI

**RAG-powered PubMed research assistant** — Ask any medical research question and get answers grounded in real peer-reviewed literature, with citations.

Built by **Sri Vardhini Venna** | MS Engineering Data Science · University of Houston

---

## What It Does

1. **Fetches** real abstracts from PubMed (NIH) on any medical topic
2. **Embeds** them into a ChromaDB vector database using sentence-transformers
3. **Retrieves** the most semantically relevant papers for your question
4. **Answers** using Mistral-7B-Instruct with citations back to source papers

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Mistral-7B-Instruct (HuggingFace free API) |
| Vector DB | ChromaDB (local persistent) |
| Embeddings | all-MiniLM-L6-v2 (CPU, no GPU needed) |
| Data Source | PubMed via NCBI E-utilities (free) |
| Frontend | Streamlit |
| Env | Python 3.10+ |

---

## Setup (5 minutes)

### 1. Clone and install
```bash
git clone https://github.com/SriVardhini-Venna/medquery-ai
cd medquery-ai
pip install -r requirements.txt
```

### 2. Get a free HuggingFace token
- Go to https://huggingface.co/settings/tokens
- Click **New token** → name it anything → select **Inference** permission
- Copy the token (starts with `hf_`)

### 3. Set up environment
```bash
cp .env.example .env
# Open .env and paste your token:
# HF_TOKEN=hf_your_token_here
```

### 4. Run
```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Usage

1. In the **sidebar**, pick a preset topic or type your own (e.g. `"sepsis deep learning"`)
2. Click **Fetch & Index Papers** — pulls 10-50 papers from PubMed
3. Type your research question in the main area
4. Click **Ask MedQuery AI** — get a cited, evidence-based answer in seconds

---

## Project Structure

```
medquery-ai/
├── app.py                  # Streamlit UI
├── rag/
│   ├── pubmed_fetcher.py   # NCBI E-utilities API wrapper
│   ├── embedder.py         # ChromaDB + sentence-transformers
│   └── llm.py              # Mistral via HuggingFace / Mistral API
├── chroma_db/              # Auto-created — persists your paper index
├── requirements.txt
├── .env.example
└── README.md
```

---

## Why This Project

This project was built to demonstrate:
- **Retrieval-Augmented Generation (RAG)** pipeline from scratch
- **Vector database** integration for semantic search (ChromaDB + FAISS-style cosine similarity)
- **LLM prompt engineering** for factual, citation-grounded medical Q&A
- **Real-world API integration** (NCBI PubMed, HuggingFace Inference)
- **Healthcare AI** application — one of the fastest-growing fields in tech

---

## License
MIT
