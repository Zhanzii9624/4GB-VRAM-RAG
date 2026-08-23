"""
scripts/download_model.py
從 Hugging Face 下載 Qwen2.5-3B-Instruct-Q4_K_M.gguf 模型檔案至 models/ 目錄。
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"
TARGET_FILENAME = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"

MODELS_DIR = Path(__file__).parent.parent / "models"


def download_model(models_dir: Path | None = None) -> Path:
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    target_path = models_dir / TARGET_FILENAME

    if target_path.exists():
        print(f"[download_model] 模型檔案已存在: {target_path}")
        return target_path

    print(f"[download_model] 開始從 Hugging Face 下載 {REPO_ID} ({FILENAME})...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=models_dir,
        local_dir_use_symlinks=False,
    )
    
    # 確保檔名一致
    downloaded = Path(downloaded_path)
    if downloaded.name != TARGET_FILENAME:
        downloaded.rename(target_path)
        
    print(f"[download_model] 下載完成，儲存至: {target_path}")
    return target_path


if __name__ == "__main__":
    download_model()
