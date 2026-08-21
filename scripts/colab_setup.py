# AORUS MASTER 16 AM6H RAG System — Colab Notebook
# ================================================================
# 此檔案為說明用的 Python 腳本版本
# 實際 Colab notebook 請見 notebook.ipynb（需在 Colab 中執行）
#
# Colab 執行流程：
#   Cell 1: 安裝環境
#   Cell 2: Clone repo / mount Drive
#   Cell 3: 安裝 llama-cpp-python (CUDA)
#   Cell 4: 下載模型
#   Cell 5: 建立 Pipeline
#   Cell 6: 互動問答
#   Cell 7: Benchmark 評測

# ── Cell 1: Install dependencies ─────────────────────────────────
INSTALL_CMD = """
# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 同步環境
uv sync

# llama-cpp-python with CUDA (T4 GPU in Colab)
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
"""

# ── Cell 2: Clone repo ────────────────────────────────────────────
CLONE_CMD = """
git clone https://github.com/YOUR_USERNAME/aorus-rag.git
cd aorus-rag
"""

# ── Cell 3: Download model ────────────────────────────────────────
DOWNLOAD_CMD = """
uv run python scripts/download_model.py
"""

# ── Cell 4: Build pipeline ────────────────────────────────────────
BUILD_CMD = """
# 解析規格（抓網頁或用備份 HTML）
uv run python rag/parser.py

# 建立 chunks
uv run python rag/chunker.py

# 計算 embeddings
uv run python rag/embedding.py
"""

# ── Cell 5: Interactive QA ────────────────────────────────────────
QA_CMD = """
uv run python main.py --query "BZH 型號的顯示晶片規格是什麼？"
"""

# ── Cell 6: Benchmark ─────────────────────────────────────────────
BENCH_CMD = """
uv run python evaluation/benchmark.py
"""

if __name__ == "__main__":
    print("請在 Colab 中執行 notebook.ipynb")
    print("或依序執行上方各步驟")
