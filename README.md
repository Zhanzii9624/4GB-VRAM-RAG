<div align="center">

# AORUS MASTER 16 AM6H 規格問答 RAG
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16hpYw6EYKeRxDs50lLdQRK_JEe4eNKtp?usp=sharing)

</div>

[GIGABYTE AORUS MASTER 16 AM6H](https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp) 的規格問答系統，於4GB VRAM限制下使用Python運行RAG pipeline (無 LangChain/LlamaIndex)

支援繁中、英文、中英混合提問，有拒答機制。

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

# 評測
uv run python scripts/eval_benchmark.py            # latency / TPS
uv run python scripts/eval_benchmark.py --ablation # retrieval 策略比較
```

Colab 連結如上，包含Drive下載模型、llama-cpp-python CUDA 安裝與eval。

---

## 模型選擇

**Qwen2.5-3B-Instruct Q4_K_M** 推論使用GPU

| | |
|---|---|
| 模型大小 | ~2.3 GB |
| 量化 | Q4_K_M（比 Q4_0 品質好，比 Q5_K_M 省 ~15% VRAM） |
| 實測 VRAM（T4, n_ctx=2048）| ~2.8 GB |
| 實測 TPS | 63.5（54.3–66.9） |

繁中能力在 3B 量化模型裡算好的，Q4_K_M 可符合4GB VRAM內。

**multilingual-e5-small** embedding 使用CPU，不占用VRAM，100+ 語言，384-dim。

---

## 架構 / Retrieval

```
parser.py → chunker.py → embedding.py → hybrid_retriever.py → prompt.py → llama_engine.py
```

- 檢索用 `score = 0.6 × cosine + 0.4 × keyword_overlap`
- keyword 用 jieba 斷詞
- 當 query 同時出現 2 個以上型號名稱(如BZH、BYH差在哪)時，純語意分數會偏向共用規格、會漏掉型號差異，故設計為遇到比較查詢時，應型號的 chunk 釘進 context 前排，alpha不影響

---

## 評測結果

**定量**

| 指標 | 平均 | 範圍 |
|------|------|------|
| Embed latency | 35.6 ms | 29.6–42.4 ms |
| Retrieval latency | 34.8 ms | 28.4–40.1 ms |
| Prefill (TTFT) | 248.6 ms | 181.4–303.4 ms |
| TPS | 63.5 | 54.3–66.9 |

15題 × 3次平均，Colab T4

**Retrieval ablation**

| 設定 | hit@3 | avg_rank |
|---|---|---|
| vector-only | 8/11 (73%) | 1.0 |
| keyword-only | 10/11 (91%) | 1.1 |
| hybrid (α=0.6) | 10/11 (91%) | 1.0|

排除 cross_variant/abstain 後的 11 題單一規格查詢，TOP_K=3
1. vector-only 無法透過語意相似度抓到精確數字
2. 此情況hybrid與 keyword-only 命中率相同

**定性測試**：15 題全對，拒答2/2

* **規格問題（BZH / BYH / BXH）**
* **問題：** 回答沒有提到規格差異，或將顯示晶片規格誤植
* **解法：** Prompt新增了以型號為單位輸出之限制，並於Retriever部分將pinned chunks依型號排序。
* **結果：** 規格與型號可完好對齊
* **BXH**：RTX 5070 Ti / 12GB / 140W
* **BYH**：RTX 5080 / 16GB / 175W
* **BZH**：RTX 5090 / 24GB / 175W


* **拒答判斷**
* **問題：** 檢索僅含「臉部辨識」時，模型誤判為支援「指紋辨識」
* **解法：** 補充Prompt 相近但非目標功能視同查無資料。
* **結果：** 正確拒答
