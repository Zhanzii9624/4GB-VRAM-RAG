"""
rag/prompt.py
純手寫 Prompt 拼接（不使用任何 template library）。

設計重點：
  - System prompt 明確限制 LLM 只能根據 context 作答
  - 若 context 無相關資訊，要求回覆固定的拒答語
  - 支援 Qwen2.5-Instruct 的 ChatML 格式
"""

from __future__ import annotations

SYSTEM_PROMPT = """你是 GIGABYTE AORUS MASTER 16 AM6H 筆電的專業規格查詢助理。

規則：
1. 只根據下方「參考資料」中的內容回答問題，不要使用你自己的知識或猜測。
2. 如果「參考資料」中沒有足夠的資訊回答問題，請明確回覆：「官方規格資料中沒有找到相關資訊，建議您至 GIGABYTE 官網確認。」
3. 回答時請引用具體的規格數值，不要含糊帶過。
4. 如果問題涉及不同型號（BZH / BYH / BXH），請分別說明各型號的差異。
5. 使用繁體中文回答，若使用者用英文提問則以英文回答。

You are a professional spec-query assistant for the GIGABYTE AORUS MASTER 16 AM6H laptop.

Rules:
1. Answer ONLY based on the "Reference Data" provided below. Do not use your own knowledge or make guesses.
2. If the reference data does not contain sufficient information, respond: "No relevant information found in the official spec data. Please check GIGABYTE's official website."
3. Cite specific spec values in your answer.
4. If the question involves multiple variants (BZH / BYH / BXH), explain differences per variant.
5. Reply in the same language as the user's question (Traditional Chinese or English).
"""


def build_prompt(
    user_query: str,
    context_chunks: list[str],
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    組裝 ChatML 格式 prompt（Qwen2.5-Instruct 格式）。

    格式：
      <|im_start|>system
      {system}
      <|im_end|>
      <|im_start|>user
      參考資料：
      {context}

      問題：{query}
      <|im_end|>
      <|im_start|>assistant
    """
    context_text = "\n\n".join(
        f"[資料 {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = (
        f"<|im_start|>system\n{system_prompt.strip()}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"參考資料：\n{context_text}\n\n"
        f"問題：{user_query}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt


def build_prompt_plaintext(
    user_query: str,
    context_chunks: list[str],
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """
    純文字格式 prompt（給不支援 ChatML 的模型使用）。
    """
    context_text = "\n\n".join(
        f"[資料 {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    prompt = (
        f"{system_prompt.strip()}\n\n"
        f"=== 參考資料 ===\n{context_text}\n\n"
        f"=== 問題 ===\n{user_query}\n\n"
        f"=== 回答 ===\n"
    )
    return prompt


if __name__ == "__main__":
    # Quick preview
    sample_chunks = [
        "[類別: 顯示晶片] [Variant: BZH]\n顯示晶片 (GPU / Graphics): NVIDIA GeForce RTX 5090 Laptop GPU, 24GB GDDR7, 175W TGP",
        "[類別: 記憶體] [Variant: 全部型號 (BZH / BYH / BXH)]\n記憶體 (Memory / RAM): 32GB DDR5 5600MHz (Max 64GB)",
    ]
    print(build_prompt("BZH 的顯示晶片規格為何？", sample_chunks))
