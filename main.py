"""
main.py
本機 CLI 入口（Colab 請使用 notebook.ipynb）。
互動式問答，支援 streaming 輸出。

用法：
  uv run python main.py
  uv run python main.py --query "BZH 的 GPU 規格是什麼？"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def build_pipeline(model_path: str | None = None):
    from rag.chunker import load_chunks, build_chunks, save_chunks
    from rag.embedding import Embedder, build_index
    from rag.retriever import build_retriever
    from inference.llama_engine import LlamaEngine
    import numpy as np

    # Chunks
    chunks_path = ROOT / "data" / "processed" / "chunks.json"
    if chunks_path.exists():
        chunks = load_chunks()
    else:
        print("[main] Building chunks from specs (first run)...")
        from rag.parser import build_specs
        specs = build_specs()
        chunks = build_chunks(specs)
        save_chunks(chunks)

    # Embeddings
    embedder = Embedder(device="cpu")
    emb_path = ROOT / "data" / "embeddings" / "chunk_embeddings.npy"
    if emb_path.exists():
        embeddings = embedder.load_embeddings()
    else:
        print("[main] Computing embeddings (first run)...")
        embedder, embeddings = build_index(chunks, embedder)

    retriever = build_retriever(chunks, embeddings, embedder)
    engine = LlamaEngine(model_path=model_path or (ROOT / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"))
    return retriever, engine


def ask(query: str, retriever, engine, top_k: int = 5) -> None:
    from rag.prompt import build_prompt

    print(f"\n{'='*60}")
    print(f"Q: {query}")
    print(f"{'─'*60}")

    retrieved = retriever.retrieve(query, top_k=top_k)
    context = [r.chunk.text for r in retrieved]
    prompt = build_prompt(query, context)

    print("A: ", end="", flush=True)
    for token in engine.stream(prompt, max_new_tokens=400):
        print(token, end="", flush=True)
    print()


def interactive_loop(retriever, engine):
    print("\n🔍 AORUS MASTER 16 AM6H 規格問答系統")
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
    parser = argparse.ArgumentParser(description="AORUS RAG QA System")
    parser.add_argument("--query", type=str, default=None, help="單次查詢（不進入互動模式）")
    parser.add_argument("--model", type=str, default=None, help="GGUF 模型路徑")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever, engine = build_pipeline(args.model)

    if args.query:
        ask(args.query, retriever, engine, top_k=args.top_k)
    else:
        interactive_loop(retriever, engine)


if __name__ == "__main__":
    main()
