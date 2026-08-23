"""
Hybrid Retrieval: cosine similarity + jieba keyword overlap, min-max normalized and fused.

比較查詢（query 同時含2+個型號）時，把所有 variant-specific chunks 強制釘入 context 前排，
避免 semantic score 偏向 shared chunks 而漏掉差異規格。
"""
import re
import numpy as np
import jieba

_ALL_VARIANTS = {"BZH", "BYH", "BXH"}


def _detect_all_variants(query: str) -> set[str]:
    q = query.upper()
    return {v for v in _ALL_VARIANTS if v in q}


def _is_comparison_query(query: str) -> bool:
    return len(_detect_all_variants(query)) >= 2


def _tokenize(text: str) -> list[str]:
    """jieba 斷詞 + 英數 regex，回傳 token list。"""
    tokens = []
    for seg in jieba.cut(text.lower()):
        seg = seg.strip()
        if seg:
            tokens.extend(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", seg))
    return tokens


def _keyword_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    if not query_tokens:
        return 0.0
    chunk_set = set(chunk_tokens)
    return sum(1 for t in query_tokens if t in chunk_set) / len(query_tokens)


def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def _get_chunk_variant(chunk) -> str:
    if isinstance(chunk, dict):
        return chunk.get("metadata", {}).get("variant", "") or chunk.get("variant", "")
    return (getattr(chunk, "metadata", {}) or {}).get("variant", "")


class HybridRetriever:
    def __init__(self, chunks, embeddings, embedder, alpha: float = 0.6):
        self.chunks = chunks
        self.embeddings = embeddings
        self.embedder = embedder
        self.alpha = alpha
        self.chunk_tokens = [
            _tokenize(c["text"] if isinstance(c, dict) else str(c))
            for c in chunks
        ]

    def _pinned_variant_chunks(self, variants: set[str]) -> list[int]:
        """找出 variant 在指定集合中、且非 shared 的所有 chunk index。"""
        return [
            i for i, chunk in enumerate(self.chunks)
            if _get_chunk_variant(chunk) in variants
            and _get_chunk_variant(chunk) != "shared"
        ]

    def search(self, query: str, top_k: int = 3):
        # vector score
        q_vec = self.embedder.model.encode([query], normalize_embeddings=True)[0]
        v_scores = self.embeddings @ q_vec

        # keyword score
        q_tokens = _tokenize(query)
        k_scores = np.array([_keyword_score(q_tokens, ct) for ct in self.chunk_tokens])

        # fusion
        final = self.alpha * _minmax_normalize(v_scores) + (1 - self.alpha) * _minmax_normalize(k_scores)

        # 比較查詢：先 pin variant-specific chunks，再補充到 top_k
        pinned: list[int] = []
        if _is_comparison_query(query):
            pinned = self._pinned_variant_chunks(_detect_all_variants(query))
            pinned.sort(key=lambda i: _get_chunk_variant(self.chunks[i]))  # 依型號排序，同型號的 chunk 在 context 裡連續出現
            if pinned:
                print(f"[retriever] comparison query → pin {len(pinned)} variant chunks: "
                    f"{[_get_chunk_variant(self.chunks[i]) for i in pinned]}")

        seen: set[int] = set()
        final_idx: list[int] = []
        for i in pinned:
            if i not in seen:
                final_idx.append(i)
                seen.add(i)
        for i in np.argsort(-final).tolist():
            if len(final_idx) >= top_k:
                break
            if i not in seen:
                final_idx.append(i)
                seen.add(i)

        return [
            {
                "chunk": self.chunks[i],
                "vector_score": float(v_scores[i]),
                "keyword_score": float(k_scores[i]),
                "final_score": float(final[i]),
            }
            for i in final_idx
        ]


def build_hybrid_retriever(chunks, embeddings, embedder, alpha: float = 0.6):
    return HybridRetriever(chunks, embeddings, embedder, alpha=alpha)
