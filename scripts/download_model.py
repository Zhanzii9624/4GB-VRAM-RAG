"""
scripts/download_model.py
下載 Qwen2.5-3B-Instruct Q4_K_M GGUF 模型到 models/ 目錄。

用法：
  python scripts/download_model.py

模型選擇理由：
  - Qwen2.5-3B-Instruct：原生支援繁中/英，3B 參數在 Q4_K_M 量化後約 2.3GB
  - Q4_K_M：量化品質與檔案大小的最佳平衡點
  - 實測 VRAM：model weights (~2.3GB) + KV cache (n_ctx=2048, ~0.3GB) ≈ 2.6GB
    → 4GB VRAM 環境下有約 1.4GB buffer（詳見 README 評測結果）
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_DIR = Path(__file__).parent.parent / "models"
TARGET = MODEL_DIR / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"


def download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: uv sync")
        sys.exit(1)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET.exists():
        size_gb = TARGET.stat().st_size / 1e9
        print(f"[download] Model already exists: {TARGET} ({size_gb:.2f} GB)")
        return TARGET

    print(f"[download] Downloading {FILENAME} from {REPO_ID} ...")
    print(f"[download] Target: {TARGET}")
    print(f"[download] This may take a few minutes (~2.3 GB) ...")

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )

    # rename to our canonical name if needed
    src = Path(path)
    if src != TARGET:
        src.rename(TARGET)

    print(f"[download] Saved → {TARGET}")
    size_gb = TARGET.stat().st_size / 1e9
    print(f"[download] File size: {size_gb:.2f} GB")
    return TARGET


if __name__ == "__main__":
    download()
