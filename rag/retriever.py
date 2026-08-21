"""
rag/retriever.py
Hybrid retrieval：Vector（語意）+ Keyword（關鍵詞）→ RRF 融合。

設計決策：
  - 純 numpy cosine similarity（brute-force），資料量 < 100 chunks 不需 FAISS
  - Keyword retrieval：中英文混合；中文用 n-gram overlap，英文/數字用 token overlap
    （沒有引入 jieba，降低依賴；README 誠實說明此限制）
  - Variant-aware filtering：query 含型號關鍵詞時優先過濾對應 variant
  - 融合公式：score = α × sem + (1-α) × kw  （α=0.7，誠實標注為經驗值）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.chunker import Chunk
    from rag.embedding import Embedder

# ── 常數 ──────────────────────────────────────────────────────────────────────
ALPHA = 0.7          # semantic weight（經驗值，未系統性調參）
TOP_K = 5            # 預設回傳 top-k chunks
NGRAM_N = 2          # 中文 n-gram 大小

VARIANT_KEYWORDS: dict[str, str] = {
    "BZH": "BZH",
    "BYH": "BYH",
    "BXH": "BXH",
    "rtx 5090": "BZH",
    "5090": "BZH",
    "rtx 5080": "BYH",
    "5080": "BYH",
    "rtx 5070 ti": "BXH",
    "5070 ti": "BXH",
    "5070ti": "BXH",
}


@dataclass
class RetrievalResult:
    chunk: "Chunk"
    score: float
    semantic_score: float
    keyword_score: float
    rank: int


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def _detect_variant(query: str) -> str | None:
    """從 query 偵測指定型號；無則回傳 None。"""
    q_lower = query.lower()
    for kw, variant in VARIANT_KEYWORDS.items():
        if kw.lower() in q_lower:
            return variant
    return None


def _tokenize(text: str) -> set[str]:
    """
    簡易 tokenizer：
      - 英文/數字：以空白或標點切詞，轉小寫
      - 中文：取所有連續中文字的 bigram
    產出 token set，用於 keyword overlap 計分。
    """
    tokens: set[str] = set()

    # 英文 token（含數字）
    en_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    tokens.update(en_tokens)

    # 中文 n-gram
    zh_chars = re.findall(r"[\u4e00-\u9fff]+", text)
    for segment in zh_chars:
        for i in range(len(segment) - NGRAM_N + 1):
            tokens.add(segment[i : i + NGRAM_N])

    return tokens


def _keyword_score(query_tokens: set[str], chunk_text: str) -> float:
    """
    計算 query 與 chunk 的 token overlap 比例。
    score = |query_tokens ∩ chunk_tokens| / max(|query_tokens|, 1)
    """
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk_text)
    overlap = query_tokens & chunk_tokens
    return len(overlap) / len(query_tokens)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)


# ── 主要 Retriever ────────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(
        self,
        chunks: "list[Chunk]",
        embeddings: np.ndarray,
        embedder: "Embedder",
        alpha: float = ALPHA,
    ):
        self.chunks = chunks
        self.embeddings = embeddings  # shape (N, D), already L2-normalized
        self.embedder = embedder
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        variant_filter: bool = True,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieve：
          1. 偵測 variant filter
          2. 計算 semantic scores（cosine）
          3. 計算 keyword scores（token overlap）
          4. 融合 → 排序 → 回傳 top_k
        """
        # 1. Variant filter
        target_variant = _detect_variant(query) if variant_filter else None
        if target_variant:
            print(f"[retriever] Variant filter: {target_variant}")

        # 2. Semantic
        q_vec = self.embedder.embed_query(query)           # shape (D,)
        sem_scores = self.embeddings @ q_vec               # shape (N,)
        sem_norm = _normalize_scores(sem_scores)

        # 3. Keyword
        q_tokens = _tokenize(query)
        kw_raw = np.array([
            _keyword_score(q_tokens, c.text)
            for c in self.chunks
        ], dtype=np.float32)
        kw_norm = _normalize_scores(kw_raw)

        # 4. Hybrid fusion
        hybrid = self.alpha * sem_norm + (1 - self.alpha) * kw_norm

        # Variant-aware boost：若 chunk variant 與 query 指定一致，加分；
        # 若 chunk 為 shared，保留（不降分）；若為其他 variant，輕微降分
        if target_variant:
            for i, chunk in enumerate(self.chunks):
                cv = chunk.metadata.get("variant", "shared")
                if cv == target_variant:
                    hybrid[i] *= 1.2     # boost
                elif cv != "shared":
                    hybrid[i] *= 0.6     # penalize wrong variant

        # 排序
        ranked_indices = np.argsort(hybrid)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append(RetrievalResult(
                chunk=self.chunks[idx],
                score=float(hybrid[idx]),
                semantic_score=float(sem_scores[idx]),
                keyword_score=float(kw_raw[idx]),
                rank=rank,
            ))

        return results

    def retrieve_texts(self, query: str, top_k: int = TOP_K) -> list[str]:
        """簡化版：只回傳 chunk text list。"""
        return [r.chunk.text for r in self.retrieve(query, top_k)]


def build_retriever(
    chunks: "list[Chunk]",
    embeddings: np.ndarray,
    embedder: "Embedder",
    alpha: float = ALPHA,
) -> HybridRetriever:
    return HybridRetriever(chunks, embeddings, embedder, alpha)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag.chunker import load_chunks
    from rag.embedding import Embedder, build_index

    chunks = load_chunks()
    embedder = Embedder()
    embeddings = embedder.load_embeddings()

    retriever = build_retriever(chunks, embeddings, embedder)

    test_queries = [
        "這台筆電的 GPU 是什麼？",
        "BXH variant GPU specs",
        "記憶體最大容量",
        "Wi-Fi 版本",
        "有沒有指紋辨識？",
        "BZH 跟 BXH 的顯示晶片差在哪？",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(f"  [{r.rank}] score={r.score:.4f} | {r.chunk.text[:80]}")
