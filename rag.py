from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import hnswlib

from paths import index_dir
from pdf_ingest import extract_units, chunk_text

INDEX_DIR = index_dir()
HNSW_FILE = INDEX_DIR / "hnsw.index"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
META_FILE = INDEX_DIR / "meta.json"
EMB_FILE = INDEX_DIR / "embeddings.npy"
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

@dataclass
class Chunk:
    id: int
    text: str
    source_name: str
    loc_kind: str
    loc: int

class BM25:
    def __init__(self):
        self.doc_ids, self.lens, self.tfs, self.df = [], [], [], {}
        self.N, self.avgdl = 0, 0.0

    def build(self, chunks: list[Chunk]):
        self.doc_ids, self.lens, self.tfs, self.df = [], [], [], {}
        for c in chunks:
            toks = [x.lower() for x in TOKEN_RE.findall(c.text)]
            if not toks:
                continue
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.doc_ids.append(c.id)
            self.lens.append(len(toks))
            self.tfs.append(tf)
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(self.doc_ids)
        self.avgdl = sum(self.lens) / self.N if self.N else 0.0

    def score(self, q: str):
        toks = [x.lower() for x in TOKEN_RE.findall(q)]
        if not toks or self.N == 0:
            return []
        results = []
        for i, did in enumerate(self.doc_ids):
            dl, tf, s = self.lens[i], self.tfs[i], 0.0
            for t in dict.fromkeys(toks):
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                denom = f + 1.5 * (1 - 0.75 + 0.75 * dl / (self.avgdl + 1e-9))
                s += idf * (f * 2.5 / (denom + 1e-9))
            if s > 0:
                results.append((did, s))
        return sorted(results, key=lambda x: x[1], reverse=True)

class RAG:
    def __init__(self):
        self.embedder = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.dim = None
        self.bm25 = BM25()

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embedder is None:
            from fastembed import TextEmbedding
            self.embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        vecs = list(self.embedder.embed(texts))
        mat = np.array(vecs, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return (mat / norms).astype(np.float32)

    def load(self) -> bool:
        try:
            if not all(f.exists() for f in [HNSW_FILE, CHUNKS_FILE, META_FILE, EMB_FILE]):
                return False
            meta = json.loads(META_FILE.read_text(encoding="utf-8"))
            self.dim = int(meta["dim"])
            self.chunks = []
            with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    self.chunks.append(Chunk(**json.loads(line)))
            self.embeddings = np.load(str(EMB_FILE)).astype(np.float32)
            idx = hnswlib.Index(space="cosine", dim=self.dim)
            idx.load_index(str(HNSW_FILE))
            idx.set_ef(60)
            self.index = idx
            self.bm25.build(self.chunks)
            return True
        except Exception:
            return False

    def build(self, file_paths: list[str], chunk_chars: int = 1500, overlap: int = 100):
        self.chunks = []
        for fp in file_paths:
            if not Path(fp).exists():
                continue
            for u in extract_units(fp):
                for t in chunk_text(u.text, chunk_chars, overlap):
                    self.chunks.append(Chunk(len(self.chunks), t, u.source_name, u.loc_kind, u.loc))
        if not self.chunks:
            return 0, 0

        texts = [c.text for c in self.chunks]
        batches = [self._embed(texts[i:i + 32]) for i in range(0, len(texts), 32)]
        self.embeddings = np.vstack(batches)
        self.dim = int(self.embeddings.shape[1])

        self.index = hnswlib.Index(space="cosine", dim=self.dim)
        self.index.init_index(max_elements=len(self.chunks) + 1024, ef_construction=200, M=16)
        self.index.set_ef(60)
        self.index.add_items(self.embeddings, np.arange(len(self.chunks), dtype=np.int32))

        self.bm25.build(self.chunks)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        np.save(str(EMB_FILE), self.embeddings)
        META_FILE.write_text(json.dumps({"dim": self.dim}, indent=2), encoding="utf-8")
        self.index.save_index(str(HNSW_FILE))
        return len(file_paths), len(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        if not self.index or not self.chunks:
            return []
        try:
            qv = self._embed([query])
            k = min(top_k * 3, len(self.chunks))
            labels, _ = self.index.knn_query(qv, k=k)
            vec_ids = [int(i) for i in labels[0]]
        except Exception:
            vec_ids = []

        bm_ids = [d for d, _ in self.bm25.score(query)[:top_k * 3]]
        scores = {}
        for r, i in enumerate(vec_ids, 1):
            scores[i] = scores.get(i, 0) + 1.0 / (60 + r)
        for r, i in enumerate(bm_ids, 1):
            scores[i] = scores.get(i, 0) + 1.0 / (60 + r)

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [self.chunks[i] for i, _ in merged if 0 <= i < len(self.chunks)]