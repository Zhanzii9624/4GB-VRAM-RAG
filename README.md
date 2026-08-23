# AORUS MASTER 16 AM6H 規格問答 RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zhanzii9624/4GB-VRAM-RAG/blob/main/scripts/colab.ipynb)

[GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp) 的規格問答系統，在 4GB VRAM 限制下用 Python 跑完整 RAG pipeline。支援繁中、英文、中英混合提問，有拒答機制。

---

## 架構

```
parser.py → chunker.py → embedding.py → hybrid_retriever.py → prompt.py → llama_engine.py
```

- 規格資料為手動維護（共有21筆）
- Embedding 使用 CPU（multilingual-e5-small），LLM 使用 GPU（Qwen2.5-3B Q4_K_M，llama-cpp-python）
- 無 LangChain 或 LlamaIndex

---

## 啟動

### 環境

```bash
git clone https://github.com/Zhanzii9624/4GB-VRAM-RAG.git
cd 4GB-VRAM-RAG
uv sync
```

Colab 用 `scripts/colab.ipynb` 跑，已整合 Drive 掛載、llama-cpp-python CUDA 安裝和 eval。

### llama-cpp-python（CUDA）

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
```

### 首次需下載模型與建立索引

```bash
uv run python scripts/download_model.py # 從 Hugging Face 下載 Qwen2.5 GGUF 模型
uv run python rag/parser.py             # 輸出 specs.json
uv run python rag/chunker.py            # 輸出 chunks.json
uv run python rag/embedding.py          # 輸出 chunk_embeddings.npy
```

### 問答

```bash
uv run python main.py
uv run python main.py --query "BZH 的顯示晶片規格是什麼？"
```

### Benchmark

```bash
uv run python scripts/eval_benchmark.py            # 完整 pipeline：latency/TPS + LLM 回答
uv run python scripts/eval_benchmark.py --ablation # 純 retrieval，比較三種 alpha 設定
```

---

## 模型選擇

### Qwen2.5-3B-Instruct Q4_K_M

Qwen2.5-3B-Instruct繁中能力較好、且Q4_K_M可以在4GB VRAM內使用。

| | |
|---|---|
| 模型大小 | ~2.3 GB |
| 量化 | Q4_K_M（比 Q4_0 品質好，比 Q5_K_M 省 ~15% VRAM） |
| 實測 VRAM（T4, n_ctx=2048）| ~2.8 GB |
| 實測 TPS（T4）| 59–68 tokens/s |
| Prefill / TTFT（中位數）| ~250 ms |

`n_ctx=2048` 足夠system prompt + 6 chunks + query 約 600–900 tokens

### multilingual-e5-small

跑 CPU，不占 VRAM；100+ 語言，384-dim；e5 系列需要加 `"query: "` / `"passage: "` prefix，否則 retrieval 準確度會下降。

---

## Retrieval 設計

```
score = 0.6 × cosine + 0.4 × keyword_overlap
```

keyword 用 jieba 斷詞（比純 bigram 對中文更準）。

**跨型號比較查詢修復**：
- 當query同時出現 2 個以上型號名稱（如「BZH、BYH、BXH 差在哪」），semantic score 天然偏向 shared chunks，會漏掉 GPU 差異。
- 解決方式：偵測到比較查詢後，把所有 variant-specific chunks（非 shared）強制 pin 到 context 前排，再補其餘 top-k。

**Retrieval ablation**（`evaluation/qa_testset.json` 11 題單一規格查詢，排除會被 pinning 覆蓋的 cross_variant 跟無正確 chunk 可比對的 abstain，TOP_K=3，指標為正確答案第一次被覆蓋完的排名）：

| 設定 | hit@3 | avg_rank |
|---|---|---|
| vector-only (α=1.0) | 8/11 (73%) | 1.0 |
| keyword-only (α=0.0) | 10/11 (91%) | 1.1 |
| hybrid (α=0.6) | 10/11 (91%) | **1.0** |

vector-only 漏掉的兩題（顯示器解析度 `2560x1600`、重量 `2.5 kg`）都是精確數字型 spec，純語意相似度本來就抓不準這種數字；keyword-only 命中率跟 hybrid 打平，但電池容量那題排名輸給 hybrid（keyword-only 排第 2，hybrid 排第 1）——代表 alpha=0.6 不是白加的，語意分數在部分題目確實把排名拉到更前面。三種設定都漏掉的那一題（CPU 型號）是測試資料的 keyword 打錯字，不是 retrieval 的問題（原文是 "Core Ultra 9 **Processor** 275HX"，測試題少打了 Processor）。

---

## 評測結果（15 題，Colab T4）

每題跑 3 次取平均。

| 指標 | 平均 | 範圍 |
|------|------|------|
| Embed latency | 35.6 ms | 29.6–42.4 ms |
| Retrieval latency | 34.8 ms | 28.4–40.1 ms |
| Prefill (TTFT) | 248.6 ms | 181.4–303.4 ms |
| TPS | 63.5 | 54.3–66.9 |

| ID | 問題 | 類型 | 結果 |
|----|------|------|------|
| Q01 | CPU 型號 | 單一規格 | ✅ Core Ultra 9 Processor 275HX |
| Q02 | BZH GPU 規格 | 單一規格 | ✅ RTX 5090, 24GB GDDR7, 175W |
| Q03 | BXH GPU（英文） | 單一規格 | ✅ RTX 5070 Ti, 12GB GDDR7 |
| Q04 | 記憶體上限 | 單一規格 | ✅ 64GB DDR5 5600MHz |
| Q05 | 顯示器解析度/更新率（英文） | 單一規格 | ✅ 2560x1600, 240Hz |
| Q06 | Wi-Fi 版本 | 單一規格 | ✅ WIFI 7 |
| Q07 | BZH vs BXH GPU 差異 | 跨型號 | ✅ 正確列出兩者規格差異 |
| Q08 | 三型號 GPU 全比較（英文） | 跨型號 | ⚠️ 規格數字正確，但排版把 BYH 誤植在 "RTX 5090" 標題底下（BYH 實際是 RTX 5080），格式易誤讀 |
| Q09 | 支援指紋辨識？ | 拒答 | ❌ 答成「支援指紋辨識」，實際規格只有 Windows Hello 臉部辨識——拒答失敗，把相近功能當成答案 |
| Q10 | 支援 Wi-Fi 8？（英文） | 拒答 | ✅ 正確拒答（規格是 Wi-Fi 7） |
| Q11 | 電池容量 | 單一規格 | ✅ 99Wh |
| Q12 | USB Type-A 埠數 | 單一規格 | ✅ 2 個 |
| Q13 | 重量（英文） | 單一規格 | ✅ 約 2.5 kg |
| Q14 | Thunderbolt 版本 | 單一規格 | ✅ Thunderbolt 4 / 5 |
| Q15 | BYH GPU 功耗 | 單一規格 | ✅ 175W |

正確率 **13/15**，拒答 **1/2**。

Q07/Q08 靠 `_pinned_variant_chunks` 把 GPU chunk 釘入 context 才答對。Q09 是目前唯一的真實失敗案例：retrieval 有抓到相關 chunk（Webcam/臉部辨識那筆），但 LLM 把「有提到生物辨識」直接當成「有指紋辨識」來回答，沒有意識到問題問的功能跟資料寫的功能不是同一個。已經在 `rag/prompt.py` 加一條規則明確要求「相近但不同的功能視同找不到」，下一輪 eval 要重新驗證這題有沒有修好。

---

## 已知限制

- `alpha=0.6` 沒有系統性調參，是人工估的；ablation 顯示 hybrid 至少不比單一策略差，但沒有掃過完整 alpha 範圍
- 拒答機制在「問題功能跟資料相近但不同」時會失效（見 Q09），已加規則修正，尚待重新驗證
- 資料只有這一個產品頁，21 筆規格、需要手動維護
- `max_new_tokens=200` 在長答案可能截斷

---

## 目錄

```
main.py                # CLI入口 (可單次提問、互動問答與 --ablation)
rag/
  parser.py            # 手動規格資料解析，輸出specs.json
  chunker.py           # 將specs轉換為可檢索的chunk，輸出chunks.json
  embedding.py         # multilingual-e5-small 向量處理與快取(.npy)
  hybrid_retriever.py  # 向量 + jieba 關鍵字混合檢索與比較查詢釘選(pinning)
  prompt.py            # ChatML 格式 Prompt 組裝與拒答機制系統提示
inference/
  llama_engine.py      # llama-cpp-python 模型載入與 Streaming 推論引擎
evaluation/
  qa_testset.json      # 15 題測試題，含單一規格、跨型號比較與拒答測試
scripts/
  download_model.py    # 自動從 HF 下載 GGUF 模型檔至 models/
  eval_benchmark.py    # 測試script，latency/TPS 完整 pipeline + retrieval-only ablation
  colab.ipynb          # Colab執行與測試
```