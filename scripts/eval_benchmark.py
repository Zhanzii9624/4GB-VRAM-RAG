"""
系統評測：
1. 拆解量測 embedding 延遲 / retrieval 延遲 / LLM prefill 延遲 / TPS，每題跑 N 次取平均
2. 定性測試集：中英混合、跨 variant 比較、拒答測試
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
TOP_K = 6  # 從 3 調成 6：21 筆資料量小，多帶一點 context 對跨 variant 比較很重要

TEST_SET = [
    {"id": "q1", "type": "中文-一般", "question": "BZH 的顯示晶片規格是什麼？"},
    {"id": "q2", "type": "英文-一般", "question": "What is the GPU of the BZH variant?"},
    {"id": "q3", "type": "中英混合", "question": "請問 BZH 這台 laptop 的 VRAM capacity 是多少？"},
    {"id": "q4", "type": "跨variant比較", "question": "BZH、BYH、BXH 三個型號中，哪一個顯示晶片的最大功耗最高？"},
    {"id": "q5", "type": "跨variant比較", "question": "BYH 的顯示晶片 VRAM 是多少？跟 BZH 差多少？"},
    {"id": "q6", "type": "拒答測試", "question": "這台筆電螢幕支援觸控功能嗎？"},  # 規格表沒有觸控欄位
    {"id": "q7", "type": "拒答測試-無關問題", "question": "今天天氣如何？"},
    {"id": "q8", "type": "拒答測試", "question": "這台筆電的保固期限是幾年？"},  # 規格表沒有保固欄位
    {"id": "q9", "type": "精確數字查詢", "question": "175W 對應的是哪些型號？"},  # 注意：BZH 和 BYH 都是 175W
    {"id": "q10", "type": "一般查詢-英文", "question": "How much RAM (system memory) can this laptop support?"},
    {"id": "q11", "type": "一般查詢-中文", "question": "電池容量是多少？"},
    {"id": "q12", "type": "中英混合", "question": "這台 laptop 的 keyboard 有支援 RGB 嗎？"},
    {"id": "q13", "type": "中英混合", "question": "連接埠 right side規格有什麼"},
    {"id": "q14", "type": "中英混合", "question": "BZH、BYH、BXH差在哪? What's the difference?"},
    {"id": "q15", "type": "中英混合", "question": "BYH有NVIDIA GeForce RTX 5070 Ti和GPU16GB對嗎? what else special?"},
]


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000
    return result, elapsed


def run_single(question: str, embedder, retriever, engine):
    _, embed_ms = timed(embedder.model.encode, [question], normalize_embeddings=True)

    results, retrieval_ms = timed(retriever.search, question, TOP_K)
    # 這裡確保 chunk_text 是字串
    context_chunks = [r["chunk"]["text"] if isinstance(r["chunk"], dict) else str(r["chunk"]) for r in results]

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
        # 存檔前轉成純字串列表，避免 JSON serializable 錯誤
        "retrieved": context_chunks,
    }


def main():
    chunks = load_chunks()
    embedder = Embedder(device="cpu")
    embeddings = embedder.load_embeddings()
    retriever = build_hybrid_retriever(chunks, embeddings, embedder)
    engine = LlamaEngine(n_gpu_layers=-1, n_ctx=2048, verbose=False)

    all_results = []
    for case in TEST_SET:
        print(f"\n=== [{case['id']}] {case['type']}: {case['question']} ===")
        runs = [run_single(case["question"], embedder, retriever, engine) for _ in range(N_RUNS)]

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
            "question": case["question"],
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


def ablation():
    """
    Retrieval-only ablation：不跑LLM，只比較三種alpha設定下
    hybrid retriever 的 chunk 命中率。
    命中定義：retrieved chunks 中至少一筆包含 expected_keywords 的所有關鍵字。
    使用 evaluation/qa_testset.json（有 expected_keywords 欄位）。
    """
    import sys
    qa_path = Path(__file__).parent.parent / "evaluation" / "qa_testset.json"
    if not qa_path.exists():
        print(f"[ablation] 找不到 {qa_path}，請確認路徑")
        sys.exit(1)

    qa_set = json.loads(qa_path.read_text(encoding="utf-8"))
    # 只測有 expected_keywords、且不是拒答題的 case
    qa_set = [q for q in qa_set if q.get("expected_keywords") and not q.get("should_abstain", False)]

    chunks = load_chunks()
    embedder = Embedder(device="cpu")
    embeddings = embedder.load_embeddings()

    def hit(retrieved_texts: list[str], keywords: list[str]) -> bool:
        """至少一個 chunk 包含所有 expected_keywords（大小寫不分）。"""
        for text in retrieved_texts:
            t = text.lower()
            if all(kw.lower() in t for kw in keywords):
                return True
        return False

    configs = [
        (1.0, "vector-only"),
        (0.0, "keyword-only"),
        (0.6, "hybrid (α=0.6)"),
    ]

    print("\n=== Retrieval Ablation ===")
    print(f"測試集：{len(qa_set)} 題（排除拒答題）\n")

    for alpha, label in configs:
        retriever = build_hybrid_retriever(chunks, embeddings, embedder, alpha=alpha)
        hits = 0
        misses = []
        for q in qa_set:
            texts = [
                r["chunk"]["text"] if isinstance(r["chunk"], dict) else str(r["chunk"])
                for r in retriever.search(q["query"], top_k=TOP_K)
            ]
            if hit(texts, q["expected_keywords"]):
                hits += 1
            else:
                misses.append(q["id"])

        acc = hits / len(qa_set) * 100
        miss_str = f"  miss: {misses}" if misses else ""
        print(f"[{label:20s}]  {hits}/{len(qa_set)} ({acc:.0f}%){miss_str}")

    print()


if __name__ == "__main__":
    import sys
    if "--ablation" in sys.argv:
        ablation()
    else:
        main()
