"""
embedder.py
Embeds PubMed papers using sentence-transformers and stores them in ChromaDB.
Runs entirely on CPU — no GPU required.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import os

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pubmed_papers"
EMBED_MODEL = "all-MiniLM-L6-v2"   # 80MB, CPU-friendly, strong semantic quality


class MedEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    #  Ingestion                                                           #
    # ------------------------------------------------------------------ #

    def add_papers(self, papers: List[Dict]) -> Tuple[int, int]:
        """
        Embed and store papers in ChromaDB.
        Returns (new_added, already_existed).
        """
        new_count = 0
        skipped = 0

        for paper in papers:
            doc_id = f"pmid_{paper['pmid']}"

            existing = self.collection.get(ids=[doc_id])
            if existing["ids"]:
                skipped += 1
                continue

            text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            embedding = self.model.encode(text, show_progress_bar=False).tolist()

            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[
                    {
                        "pmid": paper["pmid"],
                        "title": paper["title"],
                        "authors": paper.get("authors", ""),
                        "year": paper.get("year", ""),
                        "journal": paper.get("journal", ""),
                        "keywords": paper.get("keywords", ""),
                        "url": paper.get("url", ""),
                    }
                ],
            )
            new_count += 1

        return new_count, skipped

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def query(self, question: str, n_results: int = 5) -> List[Dict]:
        """Semantic nearest-neighbour search over stored papers."""
        total = self.collection.count()
        if total == 0:
            return []

        embedding = self.model.encode(question, show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, total),
            include=["documents", "metadatas", "distances"],
        )

        papers = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            relevance = round((1 - distance) * 100, 1)  # cosine → % relevance
            papers.append(
                {
                    "text": doc,
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "year": meta.get("year", ""),
                    "journal": meta.get("journal", ""),
                    "keywords": meta.get("keywords", ""),
                    "url": meta.get("url", ""),
                    "pmid": meta.get("pmid", ""),
                    "relevance": relevance,
                }
            )
        return papers

    def count(self) -> int:
        return self.collection.count()

    def list_topics(self) -> List[str]:
        """Return unique journal names (proxy for topic diversity)."""
        if self.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        journals = sorted({m.get("journal", "") for m in all_meta if m.get("journal")})
        return journals[:20]
