# AORUS MASTER 16 AM6H 規格問答 RAG

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
uv run python scripts/eval_benchmark.py
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

---

## 評測結果（15 題，Colab T4）

每題跑 3 次取平均。

| 指標 | 平均 | 範圍 |
|------|------|------|
| Embed latency | 37.9 ms | 26.6–48.7 ms |
| Retrieval latency | 37.3 ms | 25.3–49.7 ms |
| Prefill (TTFT) | 254.5 ms | 204.2–360.2 ms |
| TPS | 63.5 | 57.7–67.7 |

| ID | 問題 | 類型 | 結果 |
|----|------|------|------|
| q1 | BZH 顯示晶片規格 | 中文單查 | ✅ RTX 5090, 24GB GDDR7, 175W |
| q2 | BZH GPU (英文) | 英文單查 | ✅ RTX 5090 Laptop GPU |
| q3 | BZH VRAM capacity | 中英混合 | ✅ 24GB GDDR7 |
| q4 | 三型號最大功耗比較 | 跨型號 | ✅ BZH 175W 最高（BXH 140W） |
| q5 | BYH vs BZH VRAM 差距 | 跨型號 | ✅ 16GB vs 24GB，差 8GB |
| q6 | 螢幕支援觸控？ | 拒答 | ✅ 正確拒答 |
| q7 | 今天天氣？ | 拒答 | ✅ 正確拒答 |
| q8 | 保固期限？ | 拒答 | ✅ 正確拒答 |
| q9 | 175W 對應哪些型號 | 精確數字 | ✅ BZH 和 BYH |
| q10 | RAM 最大容量（英文） | 英文單查 | ✅ 64GB DDR5 5600MHz |
| q11 | 電池容量 | 中文單查 | ✅ 99Wh |
| q12 | Keyboard 有 RGB？ | 中英混合 | ✅ 3-zone RGB Backlit |
| q13 | 連接埠右側規格 | 中英混合 | ✅ USB-A / TB4 / MicroSD / Audio |
| q14 | BZH、BYH、BXH 差在哪 | 跨型號 | ✅ 正確列出三者 GPU 差異 |
| q15 | BYH 是 RTX 5070 Ti + 16GB？ | 事實驗證 | ✅ 更正為 RTX 5080 + 16GB |

正確率 **15/15**，拒答 **3/3**。

q4/q5/q14 為跨型號比較題，靠 `_pinned_variant_chunks` 把三個 GPU chunk 釘入 context 才答對。舊版（純 semantic）q14 回答「三者之間沒有特別差異」。

---

## 已知限制

- `alpha=0.6` 沒有系統性調參，是人工估的
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
scripts/
  download_model.py    # 自動從 HF 下載 GGUF 模型檔至 models/
  eval_benchmark.py    # 測試script，latency/TPS與ablation測試
  qa_testset.json      # 15 題測試題，含單一規格、跨型號比較與拒答測試
  colab.ipynb          # Colab執行與測試
```

