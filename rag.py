"""RAG core: two chunking strategies, SentenceTransformers embeddings, two separate ChromaDB
collections, calibrated "I don't know" threshold, grounded generation, and a cached entry point.

Run `python rag.py` for a quick sample retrieval against both collections."""
from __future__ import annotations

import config  # noqa: F401  (must be first: sets telemetry env vars)

import json
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from cache import ResponseCache
from textutil import IDK_MESSAGE, split_sentences

ROOT = Path(__file__).parent
KB_DIR = ROOT / "kb"
CHROMA_DIR = ROOT / ".chroma"
CALIBRATION_PATH = ROOT / "calibration.json"
RECOMMENDATION_PATH = ROOT / "recommendation.json"

MODEL_NAME = "all-MiniLM-L6-v2"  # free, local SentenceTransformers model
COLLECTIONS = {"fixed": "nykaa_fixed_chunks", "sentence": "nykaa_sentence_chunks"}
FIXED_WORDS = 35     # fixed-size chunk length in words
FIXED_OVERLAP = 10   # words shared between consecutive fixed chunks
TOP_K = 4

COUNTERS = {"generation_calls": 0, "retrieval_calls": 0}
CACHE = ResponseCache()


# --------------------------------------------------------------------------- documents & chunking
def load_documents() -> List[dict]:
    docs = []
    for p in sorted(KB_DIR.glob("*.md")):
        raw = p.read_text(encoding="utf-8").strip().splitlines()
        title = raw[0].lstrip("# ").strip()
        body = " ".join(l.strip() for l in raw[1:] if l.strip())
        docs.append({"doc_id": p.stem, "title": title, "text": f"{title}. {body}"})
    return docs


def chunk_fixed(text: str, size: int = FIXED_WORDS, overlap: int = FIXED_OVERLAP) -> List[str]:
    """Fixed-size word windows with overlap."""
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_sentences(text: str) -> List[str]:
    """One chunk per sentence."""
    return split_sentences(text)


CHUNKERS = {"fixed": chunk_fixed, "sentence": chunk_sentences}


def doc_of(chunk_id: str) -> str:
    return chunk_id.split("::")[0]


# --------------------------------------------------------------------------- calibration helpers
def load_calibration() -> dict:
    if CALIBRATION_PATH.exists():
        return json.loads(CALIBRATION_PATH.read_text())
    return {}


def threshold_for(strategy: str) -> float:
    cal = load_calibration()
    if strategy not in cal:
        raise RuntimeError("No calibration found. Run `python calibrate.py` first (it measures the threshold).")
    return cal[strategy]["threshold"]


def recommended_strategy() -> str:
    if RECOMMENDATION_PATH.exists():
        return json.loads(RECOMMENDATION_PATH.read_text())["strategy"]
    return "sentence"


# --------------------------------------------------------------------------- engine
class RAGEngine:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings

        try:  # after the first download the model loads from the local cache with NO network access
            self.model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:  # first run only: one-time download of the free local model
            self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        self.docs = load_documents()
        self.titles = {d["title"].lower() for d in self.docs}
        self.collections: Dict[str, object] = {}
        self._lock = threading.Lock()
        self.build_index()

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False).tolist()

    def _chunks_for(self, strategy: str, doc: dict):
        return [(f"{doc['doc_id']}::{strategy}::{i}", c) for i, c in enumerate(CHUNKERS[strategy](doc["text"]))]

    def build_index(self) -> Dict[str, int]:
        """(Re)build both collections from scratch; each strategy gets its OWN collection."""
        counts = {}
        for strategy, name in COLLECTIONS.items():
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
            col = self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
            ids, texts, metas = [], [], []
            for d in self.docs:
                for cid, c in self._chunks_for(strategy, d):
                    ids.append(cid)
                    texts.append(c)
                    metas.append({"doc_id": d["doc_id"], "strategy": strategy})
            col.upsert(ids=ids, documents=texts, embeddings=self.embed(texts), metadatas=metas)
            self.collections[strategy] = col
            counts[strategy] = col.count()
        return counts

    def retrieve(self, query: str, strategy: str, k: int = TOP_K) -> List[dict]:
        COUNTERS["retrieval_calls"] += 1
        col = self.collections[strategy]
        res = col.query(query_embeddings=self.embed([query]), n_results=min(k, col.count()))
        hits = []
        for cid, text, dist in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
            hits.append({"chunk_id": cid, "doc_id": doc_of(cid), "text": text, "similarity": round(1.0 - dist, 4)})
        return hits

    def add_document(self, title: str, text: str) -> dict:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "doc"
        doc_id = f"added_{slug}"
        doc = {"doc_id": doc_id, "title": title.strip(), "text": f"{title.strip()}. {' '.join(text.split())}"}
        with self._lock:
            (KB_DIR / f"{doc_id}.md").write_text(f"# {doc['title']}\n\n{' '.join(text.split())}\n", encoding="utf-8")
            self.docs = [d for d in self.docs if d["doc_id"] != doc_id] + [doc]
            self.titles.add(doc["title"].lower())
            n = {}
            for strategy in COLLECTIONS:
                items = self._chunks_for(strategy, doc)
                ids, texts = [i for i, _ in items], [c for _, c in items]
                self.collections[strategy].upsert(
                    ids=ids, documents=texts, embeddings=self.embed(texts),
                    metadatas=[{"doc_id": doc_id, "strategy": strategy}] * len(ids),
                )
                n[strategy] = len(ids)
        CACHE.clear()  # cached answers may be stale once the KB changes
        return {"doc_id": doc_id, "chunks_fixed": n["fixed"], "chunks_sentence": n["sentence"]}

    # ---------------------------------------------------------------- grounded generation (MOCK_LLM)
    def _generate(self, query: str, hits: List[dict], max_sentences: int = 2) -> str:
        """Mock generation: answer ONLY with the retrieved context. The deterministic 'model' picks the
        sentences from the retrieved chunks that are most similar to the query."""
        COUNTERS["generation_calls"] += 1
        seen, sents = set(), []
        for h in hits:
            for s in split_sentences(h["text"]):
                key = s.lower().strip(". ")
                if key in seen or key in self.titles:
                    continue
                seen.add(key)
                sents.append(s)
        if not sents:
            return IDK_MESSAGE
        qv = self.embed([query])[0]
        sv = self.embed(sents)
        scored = sorted(((sum(a * b for a, b in zip(qv, v)), s) for v, s in zip(sv, sents)), reverse=True)
        return " ".join(s for _, s in scored[:max_sentences])

    def grounded_answer_uncached(self, query: str, strategy: Optional[str] = None, k: int = TOP_K) -> dict:
        strategy = strategy or recommended_strategy()
        threshold = threshold_for(strategy)
        hits = self.retrieve(query, strategy, k)
        top1 = hits[0]["similarity"] if hits else 0.0
        if top1 < threshold:
            return {"query": query, "strategy": strategy, "answer": IDK_MESSAGE, "sources": [], "grounded": False,
                    "top1_similarity": top1, "threshold": threshold, "chunks": [h["text"] for h in hits]}
        answer = self._generate(query, hits)
        sources = list(dict.fromkeys(h["doc_id"] for h in hits))
        return {"query": query, "strategy": strategy, "answer": answer, "sources": sources, "grounded": True,
                "top1_similarity": top1, "threshold": threshold, "chunks": [h["text"] for h in hits]}


_engine: Optional[RAGEngine] = None
_engine_lock = threading.Lock()


def get_engine() -> RAGEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = RAGEngine()
        return _engine


def grounded_answer(query: str, strategy: Optional[str] = None, k: int = TOP_K, use_cache: bool = True) -> dict:
    """Cached entry point (keyed by normalised query text + strategy). Returns the result dict with
    an extra "cache_hit" flag."""
    strategy = strategy or recommended_strategy()
    engine = get_engine()
    if not use_cache:
        return {**engine.grounded_answer_uncached(query, strategy, k), "cache_hit": False}
    value, hit = CACHE.get_or_compute(query, lambda: engine.grounded_answer_uncached(query, strategy, k),
                                      namespace=f"{strategy}:{k}")
    return {**value, "cache_hit": hit}


if __name__ == "__main__":
    eng = get_engine()
    print("chunks indexed:", {s: c.count() for s, c in eng.collections.items()})
    q = "How many days do I have to return a lipstick?"
    for s in COLLECTIONS:
        print(f"\n[{s}] top hits for: {q}")
        for h in eng.retrieve(q, s):
            print(f"  {h['similarity']:.3f}  {h['chunk_id']}  {h['text'][:80]}")
