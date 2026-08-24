"""
rag/embedding.py
將chunks.json轉為embedding、用以計算語意相似度
載入了輕量級模型multilingual-e5-small(使用CPU,VRAM 留給 LLM)
存成data/embeddings/chunk_embeddings.npy，下次可直接載入，加快速度

輸出 384 維、經L2 normalize做cosine similarity
e5有規定prefix： query加"query: "；chunk加"passage: "
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.chunker import Chunk

EMBEDDINGS_DIR = Path(__file__).parent.parent / "data" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "chunk_embeddings.npy"
CHUNK_TEXTS_PATH = EMBEDDINGS_DIR / "chunk_texts.npy" #備份chunk文字

MODEL_NAME = "intfloat/multilingual-e5-small"


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        print(f"[embedding] Loading model: {model_name} on {device}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        print(f"[embedding] Model loaded. Embedding dim: {self.model.get_sentence_embedding_dimension()}")

    # Public API
    def embed_chunks(self, chunks: "list[Chunk]", batch_size: int = 64) -> np.ndarray:
        # 計算所有 chunk 的 passage embedding 已L2 normalize
        texts = [f"passage: {c.text}" for c in chunks]
        return self._encode(texts, batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        # 計算單一 query 的 embedding 已L2 normalize
        return self._encode([f"query: {query}"])[0]

    # Internal
    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # Encode texts → L2-normalized embeddings
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    # Cache I/O
    def save_embeddings(self, embeddings: np.ndarray, chunks: "list[Chunk]") -> None:
        """存 embeddings（.npy）與對應的 chunk texts。"""
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        # 存text list以備 debug
        texts = np.array([c.text for c in chunks], dtype=object)
        np.save(CHUNK_TEXTS_PATH, texts, allow_pickle=True)
        print(f"[embedding] Saved {embeddings.shape[0]} embeddings → {EMBEDDINGS_PATH}")

    @staticmethod
    def load_embeddings() -> np.ndarray:
        # 載入已計算embeddings
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(f"Embeddings not found: {EMBEDDINGS_PATH}. Run build_index() first.")
        embeddings = np.load(EMBEDDINGS_PATH)
        print(f"[embedding] Loaded embeddings shape: {embeddings.shape}")
        return embeddings


def build_index(chunks: "list[Chunk]", embedder: Embedder | None = None) -> tuple[Embedder, np.ndarray]:
    if embedder is None:
        embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)
    embedder.save_embeddings(embeddings, chunks)
    return embedder, embeddings


if __name__ == "__main__":
    from rag.chunker import build_chunks, load_chunks
    from pathlib import Path

    chunks_path = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
    if chunks_path.exists():
        chunks = load_chunks()
    else:
        from rag.parser import build_specs
        specs = build_specs()
        from rag.chunker import build_chunks, save_chunks
        chunks = build_chunks(specs)
        save_chunks(chunks)

    embedder, embeddings = build_index(chunks)
    print(f"Done. Shape: {embeddings.shape}")
    # sanity check 檢查前三名是否與USB有關
    q_vec = embedder.embed_query("這台筆電有幾個 USB 連接埠？")
    sims = embeddings @ q_vec
    top3 = np.argsort(sims)[::-1][:3]
    for idx in top3:
        print(f"  sim={sims[idx]:.4f}: {chunks[idx].text[:80]}")
