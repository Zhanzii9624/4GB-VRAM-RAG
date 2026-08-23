# AORUS MASTER 16 AM6H 規格問答 RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16hpYw6EYKeRxDs50lLdQRK_JEe4eNKtp?usp=sharing)

[GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp) 的規格問答系統，在 4GB VRAM 限制下用純 Python 跑完整 RAG pipeline（無 LangChain / LlamaIndex）。支援繁中、英文、中英混合提問，有拒答機制。

---

## 啟動

```bash
git clone https://github.com/Zhanzii9624/4GB-VRAM-RAG.git
cd 4GB-VRAM-RAG
uv sync

# 建索引（首次執行）
uv run python scripts/download_model.py
uv run python rag/parser.py
uv run python rag/chunker.py
uv run python rag/embedding.py

# 問答
uv run python main.py
uv run python main.py --query "BZH 的顯示晶片規格是什麼？"

# 評測（定量 latency/TPS + retrieval ablation）
uv run python scripts/eval_benchmark.py
uv run python scripts/eval_benchmark.py --ablation
```

Colab 用 `scripts/colab.ipynb`，已整合 Drive 掛載、llama-cpp-python CUDA 安裝和 eval。

---

## 模型選擇（如何符合 4GB VRAM）

**Qwen2.5-3B-Instruct Q4_K_M**（推論，GPU）

| | |
|---|---|
| 模型大小 | ~2.3 GB |
| 量化 | Q4_K_M（比 Q4_0 品質好，比 Q5_K_M 省 ~15% VRAM） |
| 實測 VRAM（T4, n_ctx=2048）| ~2.8 GB |
| 實測 TPS | 63.5（54.3–66.9） |

繁中能力在 3B 量化模型裡算好的，Q4_K_M 是 4GB VRAM 內能塞下、品質又不至於太差的量化等級。

**multilingual-e5-small**（embedding，CPU，不占 VRAM）：100+ 語言，384-dim。

---

## 架構與 Retrieval 設計

```
parser.py → chunker.py → embedding.py → hybrid_retriever.py → prompt.py → llama_engine.py
```

檢索用 `score = 0.6 × cosine + 0.4 × keyword_overlap`（keyword 用 jieba 斷詞）。當 query 同時出現 2 個以上型號名稱時，semantic score 天然偏向共用規格、會漏掉型號差異，所以偵測到比較查詢後強制把 variant-specific chunk 釘入 context 前排，繞過 alpha 加權。

---

## 評測結果

**定量（15 題 × 3 次平均，Colab T4）**

| 指標 | 平均 | 範圍 |
|------|------|------|
| Embed latency | 35.6 ms | 29.6–42.4 ms |
| Retrieval latency | 34.8 ms | 28.4–40.1 ms |
| Prefill (TTFT) | 248.6 ms | 181.4–303.4 ms |
| TPS | 63.5 | 54.3–66.9 |

**Retrieval ablation**（排除 cross_variant/abstain 後的 11 題單一規格查詢，TOP_K=3）

| 設定 | hit@3 | avg_rank |
|---|---|---|
| vector-only | 8/11 (73%) | 1.0 |
| keyword-only | 10/11 (91%) | 1.1 |
| hybrid (α=0.6) | 10/11 (91%) | **1.0** |

vector-only 漏掉的兩題都是精確數字型 spec（解析度、重量），純語意相似度抓不準這種數字；hybrid 拿到跟 keyword-only 一樣的命中率，且平均排名更好，代表 alpha 加權有實際貢獻。

**定性測試**：15 題中正確 13 題，拒答測試 2 題中對 1 題。唯一的失敗案例是拒答機制在「問題功能跟資料相近但不同」時失效——問指紋辨識，資料只有 Windows Hello 臉部辨識，模型把兩者當成同一件事回答成「支援」。已在 `rag/prompt.py` 加規則明確要求相近但不同的功能視同找不到，待下一輪重新驗證。