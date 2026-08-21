"""
inference/llama_engine.py
llama-cpp-python 推論引擎，支援 Streaming 輸出。

模型：Qwen2.5-3B-Instruct Q4_K_M GGUF
  - 預估 VRAM：~2.3 GB（model weights）+ KV cache（n_ctx=2048 下約 0.3~0.5 GB）
  - 實測 VRAM 請見 README 評測結果

4GB VRAM 策略：
  - n_gpu_layers=-1（嘗試全部 offload）
  - n_ctx=2048（夠用於規格問答，不浪費 KV cache）
  - 若 VRAM 不足，逐步降低 n_gpu_layers（e.g. 28 → 20 → 0）
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Iterator

# 預設模型路徑（下載後放這裡）
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"


class LlamaEngine:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        n_gpu_layers: int = -1,    # -1 = 全部 offload；若 OOM 請調低
        n_ctx: int = 2048,
        n_threads: int = 4,
        verbose: bool = False,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Run: python scripts/download_model.py"
            )

        print(f"[llama_engine] Loading model: {model_path.name}")
        print(f"[llama_engine] n_gpu_layers={n_gpu_layers}, n_ctx={n_ctx}")

        from llama_cpp import Llama
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
        )
        self._n_ctx = n_ctx
        print("[llama_engine] Model ready.")

    # ── Streaming generation ───────────────────────────────────────────────────

    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,   # 低溫：規格問答希望穩定輸出
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """
        逐 token yield 生成文字（streaming）。
        呼叫方可直接 for token in engine.stream(prompt): print(token, end='', flush=True)
        """
        if stop is None:
            stop = ["<|im_end|>", "<|endoftext|>"]

        output = self.llm.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=True,
        )
        for chunk in output:
            token_text = chunk["choices"][0]["text"]
            if token_text:
                yield token_text

    # ── Non-streaming (for evaluation) ────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: list[str] | None = None,
    ) -> dict:
        """
        非串流模式，回傳包含 timing 資訊的 dict：
        {
          "text": str,
          "ttft_s": float,          # Time To First Token (秒)
          "total_s": float,         # 總生成時間
          "n_tokens": int,          # 生成 token 數
          "tps": float,             # tokens per second (decode phase)
        }
        """
        if stop is None:
            stop = ["<|im_end|>", "<|endoftext|>"]

        tokens = []
        ttft: float | None = None
        t_start = time.perf_counter()

        for token_text in self.stream(
            prompt, max_new_tokens, temperature, top_p, repeat_penalty, stop
        ):
            if ttft is None:
                ttft = time.perf_counter() - t_start
            tokens.append(token_text)

        t_total = time.perf_counter() - t_start
        full_text = "".join(tokens)
        n_tokens = len(tokens)
        decode_time = max(t_total - (ttft or 0), 1e-9)
        tps = n_tokens / decode_time if decode_time > 0 else 0.0

        return {
            "text": full_text,
            "ttft_s": ttft or 0.0,
            "total_s": t_total,
            "n_tokens": n_tokens,
            "tps": tps,
        }

    def get_vram_usage_mb(self) -> float | None:
        """
        嘗試讀取目前 GPU VRAM 使用量（MB）。
        需要 pynvml 或 torch；找不到則回傳 None。
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.used / 1024 / 1024
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / 1024 / 1024
        except Exception:
            pass
        return None


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = LlamaEngine()
    test_prompt = (
        "<|im_start|>system\n你是一個助理。\n<|im_end|>\n"
        "<|im_start|>user\n請用一句話介紹 AORUS MASTER 16。\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    print("[test] Streaming output:")
    for tok in engine.stream(test_prompt, max_new_tokens=100):
        print(tok, end="", flush=True)
    print()

    vram = engine.get_vram_usage_mb()
    if vram:
        print(f"[test] VRAM used: {vram:.1f} MB")
