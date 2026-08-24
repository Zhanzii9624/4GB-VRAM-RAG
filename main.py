"""
main.py
RAG系統 CLI，可以單次查詢、互動問答與 --ablation 檢索測試。

用法：
  uv run python main.py --query "BZH 的 GPU 規格是什麼？"
  uv run python main.py --ablation
  uv run python main.py (互動問答)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag.chunker import load_chunks, build_chunks, save_chunks
from rag.embedding import Embedder, build_index
from rag.hybrid_retriever import build_hybrid_retriever
from rag.prompt import build_prompt
from inference.llama_engine import LlamaEngine


def build_pipeline(model_path: str | Path | None = None):
    # 建立含 Chunks, Embeddings, HybridRetriever 與 LlamaEngine 的 RAG Pipeline
    # 1. Chunks
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    if chunks_path.exists():
        chunks = load_chunks()
    else:
        print("[main] 首次執行：從 specs 建立 chunks...")
        from rag.parser import build_specs
        specs = build_specs()
        chunks = build_chunks(specs)
        save_chunks(chunks)

    # 2. Embeddings & Vector Index
    embedder = Embedder(device="cpu")
    emb_path = ROOT / "data" / "embeddings" / "chunk_embeddings.npy"
    if emb_path.exists():
        embeddings = embedder.load_embeddings()
    else:
        print("[main] 首次執行：計算向量索引 (Vector Index)...")
        embedder, embeddings = build_index(chunks, embedder)

    # 3. Hybrid Retriever & LLM Engine
    retriever = build_hybrid_retriever(chunks, embeddings, embedder, alpha=0.6)

    target_model = Path(model_path) if model_path else (ROOT / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    engine = LlamaEngine(model_path=target_model, n_gpu_layers=-1, n_ctx=2048)

    return retriever, engine


def ask(query: str, retriever, engine, top_k: int = 6) -> None:
    # 檢索並用 Streaming 方式輸出 LLM 回答
    print(f"\n{'='*60}")
    print(f"問：{query}")
    print(f"{'─'*60}")

    search_results = retriever.search(query, top_k=top_k)
    context_chunks = [r["chunk"]["text"] if isinstance(r["chunk"], dict) else str(r["chunk"]) for r in search_results]

    prompt = build_prompt(query, context_chunks)

    print("答：", end="", flush=True)
    for token in engine.stream(prompt, max_new_tokens=400):
        print(token, end="", flush=True)
    print("\n")


def interactive_loop(retriever, engine):
    print("\n AORUS MASTER 16 AM6H 規格問答系統")
    print("   輸入問題，按 Enter 查詢；輸入 'q' 或 'quit' 結束\n")
    while True:
        try:
            query = input("問題> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] 結束。")
            break
        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("[main] 結束。")
            break
        ask(query, retriever, engine)


def main():
    # Command-line Argument Parser
    parser = argparse.ArgumentParser(description="AORUS RAG QA System")
    parser.add_argument("--query", type=str, default=None, help="單次查詢")
    parser.add_argument("--model", type=str, default=None, help="GGUF 模型路徑")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--ablation", action="store_true", help="執行 Retrieval Ablation 測試")
    args = parser.parse_args()

    if args.ablation:
        from scripts.eval_benchmark import ablation
        ablation()
        return

    retriever, engine = build_pipeline(args.model)

    if args.query:
        ask(args.query, retriever, engine, top_k=args.top_k)
    else:
        interactive_loop(retriever, engine)


if __name__ == "__main__":
    main()
