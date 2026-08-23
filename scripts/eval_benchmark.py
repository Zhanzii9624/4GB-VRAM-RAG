"""
系統評測：
1. 拆解量測 embedding 延遲 / retrieval 延遲 / LLM prefill 延遲 / TPS，每題跑 N 次取平均
2. 定性測試集：中英混合、跨 variant 比較、拒答測試
使用 scripts/qa_testset.json
"""
import time
import json
import statistics
from pathlib import Path

from rag.chunker import load_chunks
from rag.embedding import Embedder
from rag.hybrid_retriever import build_hybrid_retriever
from rag.prompt import build_prompt
from inference.llama_engine import LlamaEngine

N_RUNS = 3
TOP_K = 6            # main()：完整 pipeline 生成回答用，21 筆資料量小，多帶一點 context 對跨 variant 比較很重要
ABLATION_TOP_K = 3   # ablation()：只用於 alpha 比較，窗口收窄才能逼出三種策略的差異

QA_TESTSET_PATH = Path(__file__).parent / "qa_testset.json"

# ablation 只比較「單一規格查詢」的 retrieval 排名。
# cross_variant 會被 hybrid_retriever 的 pinning 機制強制覆蓋 alpha，abstain 沒有正確 chunk 可比對排名，
# 兩者都不適合拿來測 alpha 有沒有效，所以排除，但仍會印出排除清單，不是靜默跳過。
ABLATION_EXCLUDED_TYPES = {"cross_variant", "abstain"}


def _load_testset() -> list[dict]:
    return json.loads(QA_TESTSET_PATH.read_text(encoding="utf-8"))


def _chunk_text(r) -> str:
    return r["chunk"]["text"] if isinstance(r["chunk"], dict) else str(r["chunk"])


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def run_single(question: str, embedder, retriever, engine):
    _, embed_ms = timed(embedder.model.encode, [question], normalize_embeddings=True)

    results, retrieval_ms = timed(retriever.search, question, TOP_K)
    context_chunks = [_chunk_text(r) for r in results]

    prompt = build_prompt(question, context_chunks)

    start = time.perf_counter()
    first_token_time = None
    n_tokens = 0
    answer_tokens = []
    for tok in engine.stream(prompt, max_new_tokens=200):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        answer_tokens.append(tok)
        n_tokens += 1
    end = time.perf_counter()

    prefill_ms = (first_token_time - start) * 1000
    tps = n_tokens / (end - first_token_time) if first_token_time else 0.0

    return {
        "embed_ms": embed_ms,
        "retrieval_ms": retrieval_ms,
        "prefill_ms": prefill_ms,
        "tps": tps,
        "n_tokens": n_tokens,
        "answer": "".join(answer_tokens),
        "retrieved": context_chunks,
    }


def main():
    chunks = load_chunks()
    embedder = Embedder(device="cpu")
    embeddings = embedder.load_embeddings()
    retriever = build_hybrid_retriever(chunks, embeddings, embedder)
    engine = LlamaEngine(n_gpu_layers=-1, n_ctx=2048, verbose=False)

    testset = _load_testset()
    all_results = []
    for case in testset:
        print(f"\n=== [{case['id']}] {case['type']}: {case['query']} ===")
        runs = [run_single(case["query"], embedder, retriever, engine) for _ in range(N_RUNS)]

        avg = {
            "embed_ms": statistics.mean(r["embed_ms"] for r in runs),
            "retrieval_ms": statistics.mean(r["retrieval_ms"] for r in runs),
            "prefill_ms": statistics.mean(r["prefill_ms"] for r in runs),
            "tps": statistics.mean(r["tps"] for r in runs),
        }
        print(f"embedding: {avg['embed_ms']:.1f} ms | retrieval: {avg['retrieval_ms']:.1f} ms "
              f"| prefill(TTFT): {avg['prefill_ms']:.1f} ms | TPS: {avg['tps']:.2f}")
        print(f"回答: {runs[-1]['answer'][:200]}")
        print(f"檢索到 {len(runs[-1]['retrieved'])} 筆 chunk")

        all_results.append({
            "id": case["id"],
            "type": case["type"],
            "query": case["query"],
            "avg_embed_ms": avg["embed_ms"],
            "avg_retrieval_ms": avg["retrieval_ms"],
            "avg_prefill_ms": avg["prefill_ms"],
            "avg_tps": avg["tps"],
            "sample_answer": runs[-1]["answer"],
            "retrieved_chunks": runs[-1]["retrieved"],
        })

    out_path = Path("eval_results.json")
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存到 {out_path}")


def _first_hit_rank(retrieved_texts: list[str], keywords: list[str]) -> int | None:
    """依序看 top-k retrieved chunks，回傳所有 keyword 第一次被累積覆蓋完的名次（1-indexed）。
    沒有任何名次能覆蓋完全部 keyword 就回傳 None（miss）。"""
    seen = ""
    for i, text in enumerate(retrieved_texts, start=1):
        seen += text.lower()
        if all(kw.lower() in seen for kw in keywords):
            return i
    return None


def ablation():
    """
    Retrieval-only ablation：不跑 LLM，只比較三種 alpha 設定下 hybrid retriever 的排名品質。

    - TOP_K 收窄到 ABLATION_TOP_K=3（21 筆資料用 top_k=6 太寬鬆，三種策略幾乎都能命中，測不出差異）
    - 排除 cross_variant / abstain 題型（見 ABLATION_EXCLUDED_TYPES 說明），只測單一規格查詢
    - 指標改成 rank（正確答案第一次被覆蓋完的名次），而不是「top-k 裡有沒有出現過」的二元 hit，
      這樣才看得出 vector-only / keyword-only / hybrid 排序品質的真實差距
    """
    testset = _load_testset()
    graded = [q for q in testset if q["type"] not in ABLATION_EXCLUDED_TYPES]
    excluded = [q for q in testset if q["type"] in ABLATION_EXCLUDED_TYPES]

    chunks = load_chunks()
    embedder = Embedder(device="cpu")
    embeddings = embedder.load_embeddings()

    configs = [
        (1.0, "vector-only"),
        (0.0, "keyword-only"),
        (0.6, "hybrid (α=0.6)"),
    ]

    print("\n=== Retrieval Ablation ===")
    print(f"TOP_K={ABLATION_TOP_K}　測試集：{len(graded)} 題單一規格查詢")
    print(f"排除 {len(excluded)} 題（cross_variant/abstain，另外報告）："
          f"{[q['id'] for q in excluded]}\n")

    ablation_output = {
        "top_k": ABLATION_TOP_K,
        "excluded_ids": [q["id"] for q in excluded],
        "excluded_reason": "cross_variant 會被 pinning 機制覆蓋 alpha；abstain 無正確 chunk 可比對排名",
        "configs": {},
    }

    for alpha, label in configs:
        retriever = build_hybrid_retriever(chunks, embeddings, embedder, alpha=alpha)
        per_question = []
        for q in graded:
            texts = [_chunk_text(r) for r in retriever.search(q["query"], top_k=ABLATION_TOP_K)]
            rank = _first_hit_rank(texts, q["expected_keywords"])
            per_question.append({"id": q["id"], "rank": rank})

        ranks_hit = [pq["rank"] for pq in per_question if pq["rank"] is not None]
        misses = [pq["id"] for pq in per_question if pq["rank"] is None]
        hit_rate = len(ranks_hit) / len(graded) * 100
        avg_rank = round(statistics.mean(ranks_hit), 3) if ranks_hit else None

        avg_rank_str = f"{avg_rank:.2f}" if avg_rank is not None else "N/A"
        print(f"[{label:20s}] hit@{ABLATION_TOP_K}={len(ranks_hit)}/{len(graded)} "
              f"({hit_rate:.0f}%)  avg_rank={avg_rank_str}")
        if misses:
            print(f"    miss: {misses}")

        ablation_output["configs"][label] = {
            "alpha": alpha,
            "hit_rate": f"{len(ranks_hit)}/{len(graded)}",
            "hit_rate_pct": round(hit_rate, 1),
            "avg_rank": avg_rank,
            "per_question": per_question,
        }

    out_path = Path("eval_results_ablation.json")
    out_path.write_text(json.dumps(ablation_output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存到 {out_path}")


if __name__ == "__main__":
    import sys
    if "--ablation" in sys.argv:
        ablation()
    else:
        main()