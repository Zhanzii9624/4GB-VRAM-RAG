"""
evaluation/benchmark.py
系統評測腳本。

定量指標（N 次取統計量）：
  - TTFT：embedding + retrieval + LLM prefill 到第一個 token
  - TPS：生成 token 數 / decode 時間
  - VRAM：推論時同步記錄

定性評估：
  - 15 題測試集（qa_testset.json）
  - 自動判斷：expected_keywords 是否出現在回答中
  - 人工評分欄位：correct / hallucination / retrieval_hit (Y/N)

用法：
  python evaluation/benchmark.py          # 跑完整 benchmark
  python evaluation/benchmark.py --quick  # 只跑前 3 題（快速驗證）
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# 確保專案根目錄在 sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.chunker import load_chunks, build_chunks
from rag.embedding import Embedder, build_index
from rag.retriever import build_retriever
from rag.prompt import build_prompt
from inference.llama_engine import LlamaEngine

TESTSET_PATH = Path(__file__).parent / "qa_testset.json"
RESULTS_PATH = Path(__file__).parent / "benchmark_results.json"


# ── 初始化管線 ────────────────────────────────────────────────────────────────

def init_pipeline(model_path: str | None = None) -> tuple:
    """載入所有元件，回傳 (chunks, retriever, engine)。"""
    # 1. Chunks
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    if chunks_path.exists():
        chunks = load_chunks()
    else:
        print("[benchmark] Building chunks from specs...")
        from rag.parser import build_specs
        specs = build_specs()
        from rag.chunker import save_chunks
        chunks = build_chunks(specs)
        save_chunks(chunks)

    # 2. Embeddings
    embedder = Embedder(device="cpu")
    emb_path = ROOT / "data" / "embeddings" / "chunk_embeddings.npy"
    if emb_path.exists():
        import numpy as np
        embeddings = embedder.load_embeddings()
    else:
        print("[benchmark] Building embeddings...")
        embedder, embeddings = build_index(chunks, embedder)

    # 3. Retriever
    retriever = build_retriever(chunks, embeddings, embedder)

    # 4. LLM
    engine = LlamaEngine(
        model_path=model_path or (ROOT / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
    )
    return chunks, retriever, engine


# ── 單題評測 ─────────────────────────────────────────────────────────────────

def evaluate_one(
    question: dict,
    retriever,
    engine: LlamaEngine,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    執行一題並回傳評測結果 dict，包含 TTFT、TPS、VRAM、自動關鍵詞命中等。
    """
    query = question["query"]
    expected_kws = question.get("expected_keywords", [])
    should_abstain = question.get("should_abstain", False)

    result: dict[str, Any] = {
        "id": question["id"],
        "query": query,
        "lang": question.get("lang"),
        "type": question.get("type"),
        "target_variant": question.get("target_variant"),
        "should_abstain": should_abstain,
    }

    # ── Step 1: Embedding query ───────────────────────────────────────────────
    t0 = time.perf_counter()
    retrieved = retriever.retrieve(query, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    result["retrieval_ms"] = round(retrieval_ms, 2)

    # 記錄 Top-3 retrieval 命中（人工評分用）
    result["top3_chunks"] = [
        {
            "rank": r.rank,
            "score": round(r.score, 4),
            "variant": r.chunk.metadata.get("variant"),
            "category": r.chunk.metadata.get("category"),
            "text_preview": r.chunk.text[:100],
        }
        for r in retrieved[:3]
    ]

    # ── Step 2: Build prompt ──────────────────────────────────────────────────
    context_texts = [r.chunk.text for r in retrieved]
    prompt = build_prompt(query, context_texts)

    # ── Step 3: Generate（含 TTFT 計時）──────────────────────────────────────
    t_gen_start = time.perf_counter()
    gen = engine.generate(prompt, max_new_tokens=400, temperature=0.1)
    answer = gen["text"].strip()

    result["answer"] = answer
    result["ttft_s"] = round(gen["ttft_s"] + retrieval_ms / 1000, 4)  # 含 retrieval
    result["total_s"] = round(gen["total_s"], 4)
    result["n_tokens"] = gen["n_tokens"]
    result["tps"] = round(gen["tps"], 2)

    # VRAM
    vram = engine.get_vram_usage_mb()
    result["vram_mb"] = round(vram, 1) if vram else None

    # ── Step 4: Auto-scoring ──────────────────────────────────────────────────
    answer_lower = answer.lower()
    kw_hits = [kw for kw in expected_kws if kw.lower() in answer_lower]
    kw_miss = [kw for kw in expected_kws if kw.lower() not in answer_lower]

    result["kw_hit_count"] = len(kw_hits)
    result["kw_total"] = len(expected_kws)
    result["kw_hits"] = kw_hits
    result["kw_misses"] = kw_miss
    result["auto_correct"] = len(kw_miss) == 0 and len(expected_kws) > 0

    # 拒答檢測（看回答是否包含拒答語）
    abstain_signals = [
        "沒有找到", "no relevant information", "official spec", "官方規格資料中沒有",
        "please check gigabyte", "建議您至", "找不到"
    ]
    result["did_abstain"] = any(sig.lower() in answer_lower for sig in abstain_signals)
    result["abstain_correct"] = (result["did_abstain"] == should_abstain)

    # 人工評分欄位（預設 None，讓人工填入）
    result["human_correct"] = None        # Y/N/Partial
    result["human_hallucination"] = None  # Y/N
    result["retrieval_hit_yn"] = None     # Y/N (Top-3 有沒有抓到正確 chunk)

    return result


# ── 統計摘要 ─────────────────────────────────────────────────────────────────

def summarize(results: list[dict]) -> dict:
    ttfts = [r["ttft_s"] for r in results if r.get("ttft_s") is not None]
    tpss = [r["tps"] for r in results if r.get("tps") is not None]
    vrams = [r["vram_mb"] for r in results if r.get("vram_mb") is not None]

    auto_correct = sum(1 for r in results if r.get("auto_correct"))
    abstain_correct = sum(1 for r in results if r.get("abstain_correct"))
    abstain_qs = [r for r in results if r.get("should_abstain")]

    def stats(xs: list[float]) -> dict:
        if not xs:
            return {}
        return {
            "mean": round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3),
            "min": round(min(xs), 3),
            "max": round(max(xs), 3),
            "stdev": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
        }

    return {
        "n_questions": len(results),
        "auto_correct_rate": f"{auto_correct}/{len(results)} ({100*auto_correct/len(results):.1f}%)",
        "abstain_correct_rate": f"{abstain_correct}/{len(abstain_qs)} ({100*abstain_correct/max(len(abstain_qs),1):.1f}%)",
        "ttft_s": stats(ttfts),
        "tps": stats(tpss),
        "vram_mb": stats(vrams) if vrams else "N/A (pynvml not available)",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark")
    parser.add_argument("--quick", action="store_true", help="只跑前 3 題")
    parser.add_argument("--model", type=str, default=None, help="GGUF 模型路徑")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top-k")
    args = parser.parse_args()

    testset = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    if args.quick:
        testset = testset[:3]
        print(f"[benchmark] Quick mode: running {len(testset)} questions")
    else:
        print(f"[benchmark] Running {len(testset)} questions")

    chunks, retriever, engine = init_pipeline(args.model)

    results = []
    for i, question in enumerate(testset, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(testset)}] Q{question['id']}: {question['query']}")
        try:
            r = evaluate_one(question, retriever, engine, top_k=args.top_k)
            results.append(r)
            print(f"  TTFT: {r['ttft_s']}s | TPS: {r['tps']} | auto_correct: {r['auto_correct']}")
            print(f"  Answer preview: {r['answer'][:120]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": question["id"], "error": str(e)})

    # 摘要
    print(f"\n{'='*60}")
    print("SUMMARY")
    summary = summarize([r for r in results if "error" not in r])
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 儲存結果
    output = {"summary": summary, "results": results}
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[benchmark] Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
