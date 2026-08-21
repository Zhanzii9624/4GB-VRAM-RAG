# AORUS MASTER 16 AM6H — RAG 規格問答系統

> RAG pipeline，在 4GB VRAM 限制下精準回答 [GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp) 產品規格。
> 支援繁體中文與英文混合提問，具備 Streaming 輸出與拒答機制。

---

## 架構概覽

```
網頁規格表（HTML）
      │
      ▼
┌─────────────┐    specs.json     ┌─────────────┐   chunks.json
│  Parser     │ ────────────────► │  Chunker    │ ──────────────►
│(parser.py)  │                   │(chunker.py) │
└─────────────┘                   └─────────────┘
                                                        │
                                    ┌───────────────────┘
                                    ▼
                          ┌─────────────────────┐
                          │  Embedder           │  multilingual-e5-small
                          │  (embedding.py)     │  CPU only, 384-dim
                          └─────────────────────┘
                                    │ chunk_embeddings.npy
                                    ▼
User Query ──► embed_query ──► ┌──────────────────────┐
                               │  HybridRetriever     │
                               │  (retriever.py)      │
                               │  Vector (cosine) +   │
                               │  Keyword (n-gram)    │
                               │  Variant-aware boost │
                               └──────────────────────┘
                                    │ Top-5 chunks
                                    ▼
                          ┌─────────────────────┐
                          │  Prompt Builder     │  ChatML format
                          │  (prompt.py)        │  + 拒答 system prompt
                          └─────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────────┐
                          │  LlamaEngine        │  Qwen2.5-3B Q4_K_M
                          │  (llama_engine.py)  │  llama-cpp-python
                          │  Streaming output   │  n_gpu_layers=-1
                          └─────────────────────┘
```

**Hard constraints 遵守狀況：**
- ✅ 無 LangChain / LlamaIndex（純手寫 Python）
- ✅ 環境管理：uv
- ✅ 推論引擎：llama-cpp-python（llama.cpp binding）
- ✅ 繁中/英混合支援

---

## 啟動步驟

### 1. 環境準備

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/aorus-rag.git
cd aorus-rag

# 安裝依賴（uv 自動讀取 pyproject.toml）
uv sync
```

### 2. 安裝 llama-cpp-python（GPU 版本）

**Colab / Linux + CUDA：**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
```

**CPU only（本機測試）：**
```bash
uv pip install ".[cpu]"
```

### 3. 下載模型

```bash
uv run python scripts/download_model.py
# 下載 Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2.3 GB) 至 models/ 目錄
```

### 4. 建立資料索引（首次執行）

```bash
# 解析規格頁面（含離線備份 fallback）
uv run python rag/parser.py

# 建立 chunks
uv run python rag/chunker.py

# 計算並快取 embeddings
uv run python rag/embedding.py
```

### 5. 啟動問答

```bash
# 互動模式
uv run python main.py

# 單次查詢
uv run python main.py --query "BZH 的顯示晶片規格是什麼？"
uv run python main.py --query "What GPU does the BXH variant have?"
```

### 6. 執行 Benchmark

```bash
uv run python evaluation/benchmark.py
# 快速模式（只跑前 3 題）
uv run python evaluation/benchmark.py --quick
```

---

## 模型選擇理由

### 語言模型：Qwen2.5-3B-Instruct Q4_K_M

| 指標 | 數值 |
|------|------|
| 模型參數 | 3B |
| 量化方式 | Q4_K_M（4-bit，K-quant，Medium size） |
| 模型檔案大小 | ~2.3 GB |
| 實測 VRAM（n_ctx=2048）| **待補（Colab 實測後填入）** MB |
| 實測 VRAM（n_ctx=4096）| **待補** MB |

**選擇原因：**

1. **繁中支援**：Qwen2.5 系列以繁中/簡中能力著稱，對中文規格詞彙（「顯示晶片」、「記憶體」等）有良好的理解與生成能力。

2. **4GB VRAM 相容性**：
   - Q4_K_M 量化：每個參數約用 4.5 bits，3B 模型 ≈ 1.68 GB（純參數）
   - 加上 KV cache（n_ctx=2048）約 0.3~0.5 GB
   - 實測總 VRAM ≈ **待補（Colab 跑完後填入）** GB
   - 保守估計在 4GB VRAM 下有 1+ GB buffer
   - ⚠️ 以上為估算，**以 `nvidia-smi` 實測值為準**（見評測結果）

3. **Q4_K_M vs 其他量化**：
   - Q4_K_M 在 4-bit 量化中屬品質較高的方案（K-quant 使用混合精度）
   - 相比 Q4_0 困惑度損失更小，相比 Q5_K_M 節省約 15% VRAM
   - 對規格問答（封閉域）效果明顯優於 Q2/Q3 量化

4. **Context length 策略**：
   - 設定 `n_ctx=2048`：規格問答 prompt（system + 5 chunks + query）約 500~800 tokens，2048 已充裕
   - 不開大 context 是節省 VRAM 的關鍵決策

### Embedding 模型：multilingual-e5-small

| 指標 | 數值 |
|------|------|
| 模型大小 | ~120 MB |
| Embedding 維度 | 384 |
| 執行位置 | **CPU**（不占 VRAM） |
| 多語言支援 | 100+ 語言 |

**選擇原因：**

1. **CPU 執行**：保留全部 VRAM 給 LLM，embedding 在 CPU 跑（chunks 少，速度可接受）
2. **多語言**：原生支援繁中與英文混合查詢
3. **e5 prefix 規範**：
   - query 端加 `"query: "` prefix
   - passage 端加 `"passage: "` prefix
   - **沒加 prefix 會導致 retrieval 準確率明顯下降**（官方 model card 明確說明）
4. **`small` vs `large`**：small 在本任務（封閉域規格）足夠精準，large 模型對 CPU 速度影響更大

---

## 技術細節

### Chunking 設計

- **粒度**：一個 key-value 規格 = 一個 chunk（不過粗、不過細）
- **Variant 標註**：
  - `shared` variant：標註「適用於全部型號 (BZH/BYH/BXH)」，確保「BXH 的記憶體」能命中
  - variant-specific（BZH/BYH/BXH）：標註具體型號
- **雙語 key**：`顯示晶片 (GPU / Graphics):`，幫助英文查詢也能命中

### Hybrid Retrieval 設計

```
score = 0.7 × semantic_score + 0.3 × keyword_score
```

- **Semantic**：multilingual-e5-small cosine similarity（純 numpy，brute-force）
- **Keyword**：token overlap（英文空白切詞 + 中文 bigram）
  - ⚠️ **已知限制**：中文 keyword retrieval 效果有限（無 jieba 分詞），中文查詢主要依賴 semantic retrieval
- **Variant boost**：query 含型號關鍵詞時，對應 variant chunk 分數 ×1.2，其他 variant ×0.6
- **α=0.7** 為經驗值，未做系統性 grid search（誠實說明）

### 拒答機制

System prompt 明確指示：
> 「如果參考資料中沒有足夠的資訊回答問題，請明確回覆：『官方規格資料中沒有找到相關資訊，建議您至 GIGABYTE 官網確認。』」

測試案例：
- ❌ 「有沒有指紋辨識？」→ 規格只有 Windows Hello Face + TPM，沒有指紋辨識
- ❌ 「支援 Wi-Fi 8 嗎？」→ 規格是 Wi-Fi 7，應拒答

---

## 評測結果

> ⚠️ 以下數字為**待補**欄位，需在 Colab 實際執行後填入。

### 定量指標（N=15 題平均）

| 指標 | 平均 | 中位數 | 最小 | 最大 |
|------|------|--------|------|------|
| TTFT (s) | **待補** | **待補** | **待補** | **待補** |
| TPS (tokens/s) | **待補** | **待補** | **待補** | **待補** |
| VRAM 使用量 (MB) | **待補** | — | — | — |

**TTFT 拆解：**
- Embedding query：~X ms
- Retrieval（numpy cosine，N chunks）：~X ms
- Prompt 組裝：<1 ms
- LLM prefill（到第一個 token）：~X ms
- **總 TTFT：~X s**

### 定性 Benchmark（15 題）

| 問題 ID | 問題（摘要）| 類型 | 語言 | 自動正確 | 人工評分 | 有無幻覺 |
|---------|------------|------|------|---------|---------|---------|
| Q01 | CPU 型號 | single_spec | zh | **待補** | **待補** | **待補** |
| Q02 | BZH GPU 規格 | single_spec | zh | **待補** | **待補** | **待補** |
| Q03 | BXH GPU (英文) | single_spec | en | **待補** | **待補** | **待補** |
| Q04 | 記憶體上限 | single_spec | zh | **待補** | **待補** | **待補** |
| Q05 | 螢幕規格 (英文) | single_spec | en | **待補** | **待補** | **待補** |
| Q06 | Wi-Fi 版本 | single_spec | zh | **待補** | **待補** | **待補** |
| Q07 | BZH vs BXH GPU | cross_variant | zh | **待補** | **待補** | **待補** |
| Q08 | 三型號 GPU 比較 | cross_variant | en | **待補** | **待補** | **待補** |
| Q09 | 有指紋辨識？ | **abstain** | zh | **待補** | **待補** | **待補** |
| Q10 | 支援 Wi-Fi 8？ | **abstain** | en | **待補** | **待補** | **待補** |
| Q11 | 電池容量 | single_spec | zh | **待補** | **待補** | **待補** |
| Q12 | USB Type-A 數量 | single_spec | zh | **待補** | **待補** | **待補** |
| Q13 | 重量 (英文) | single_spec | en | **待補** | **待補** | **待補** |
| Q14 | Thunderbolt 版本 | single_spec | zh | **待補** | **待補** | **待補** |
| Q15 | BYH GPU TGP | single_spec | zh | **待補** | **待補** | **待補** |

**整體正確率：X/15（待補）**
**拒答正確率：X/2（待補）**

---

## 已知限制與未來改進

### 已知限制

1. **中文 keyword retrieval 效果有限**
   - 未引入 jieba 分詞，中文 bigram 覆蓋不完整
   - 中文查詢主要依賴 semantic retrieval（在本任務效果已足夠）

2. **α=0.7 未系統性調參**
   - Hybrid fusion 的語意/關鍵詞權重為人工設定的經驗值
   - 理想情況應做 grid search（α ∈ {0.3, 0.5, 0.7, 0.9}）

3. **資料集範圍有限**
   - 僅涵蓋單一產品頁（AORUS MASTER 16 AM6H）
   - 未涵蓋 FAQ、用戶手冊等補充資料

4. **Colab 環境依賴**
   - llama-cpp-python CUDA 版本需在 Colab 重新編譯，首次啟動較慢（~5分鐘）

### 未來改進方向

- [ ] 引入 jieba 提升中文 keyword retrieval
- [ ] 系統性調參 α（grid search + validation set）
- [ ] 擴充 embedding 模型選項（bge-m3）
- [ ] 加入 reranker（cross-encoder）進一步提升 retrieval 精準度
- [ ] 支援多產品（擴充資料集）

---

## 目錄結構

```
├── pyproject.toml       # uv 環境設定
├── uv.lock              # 鎖定依賴版本（請務必 commit）
├── README.md
├── main.py              # CLI 互動入口
├── data/
│   ├── raw/             # 原始 HTML 備份（不依賴即時網路）
│   └── processed/       # specs.json, chunks.json
├── rag/
│   ├── parser.py        # 規格頁面解析（含離線備份）
│   ├── chunker.py       # Structured → 自然語言 chunks
│   ├── embedding.py     # multilingual-e5-small wrapper
│   ├── retriever.py     # Hybrid retrieval（vector + keyword）
│   └── prompt.py        # ChatML prompt builder + 拒答
├── inference/
│   └── llama_engine.py  # llama-cpp-python wrapper + streaming
├── evaluation/
│   ├── qa_testset.json  # 15 題測試集
│   └── benchmark.py     # 定量（TTFT/TPS/VRAM）+ 定性評測
└── scripts/
    └── download_model.py # 模型下載腳本
```

---

*建置時間：2026-08-21 | 截止：2026-08-25 18:00*
