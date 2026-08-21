"""
rag/chunker.py
將 specs.json 的每筆 record 轉為可檢索的自然語言 chunk。

Chunk 格式（固定模板，方便除錯與重現）：
────────────────────────────────────────────
[類別: 顯示晶片] [Variant: BZH]
顯示晶片 (GPU / Graphics): NVIDIA GeForce RTX 5090 Laptop GPU, 24GB GDDR7, 175W TGP
產品型號 BZH 專屬規格。
────────────────────────────────────────────

shared variant 額外標註：
  「本規格適用於全部型號 (BZH / BYH / BXH)。」
以確保查「BXH 的記憶體」時也能命中 shared chunk。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
SPECS_PATH = PROCESSED_DIR / "specs.json"

ALL_VARIANTS = ("BZH", "BYH", "BXH")


@dataclass
class Chunk:
    text: str                   # 送進 embedding / retriever 的文字
    metadata: dict[str, Any]    # category, variant, key, source_url, ...

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}


def _build_chunk_text(record: dict) -> str:
    """
    將一筆 record 轉為自然語言文字。
    同時包含繁中 key 與英文 key_en，增強跨語系 retrieval。
    """
    variant = record["variant"]
    category = record["category"]
    key = record["key"]
    key_en = record.get("key_en", "")
    value = record["value"]
    note = record.get("note")

    # key 顯示：「顯示晶片 (GPU / Graphics)」
    if key_en:
        key_display = f"{key} ({key_en})"
    else:
        key_display = key

    # header line
    if variant == "shared":
        variant_label = "全部型號 (BZH / BYH / BXH)"
        variant_note = f"本規格適用於全部型號 (BZH / BYH / BXH)。"
    else:
        variant_label = variant
        variant_note = f"產品型號 {variant} 專屬規格。"

    lines = [
        f"[類別: {category}] [Variant: {variant_label}]",
        f"{key_display}: {value}",
        variant_note,
    ]

    if note:
        lines.append(f"備註: {note}")

    return "\n".join(lines)


def build_chunks(specs: dict[str, Any] | None = None) -> list[Chunk]:
    """
    從 specs dict（或從磁碟讀 specs.json）建立 chunk list。
    """
    if specs is None:
        if not SPECS_PATH.exists():
            raise FileNotFoundError(f"specs.json not found: {SPECS_PATH}. Run parser.py first.")
        specs = json.loads(SPECS_PATH.read_text(encoding="utf-8"))

    records = specs["records"]
    source_url = specs.get("source_url", "")
    product_family = specs.get("product_family", "")
    chunks: list[Chunk] = []

    # 去重 guard（parser 已去重，這裡雙保險）
    seen: set[str] = set()

    for record in records:
        uid = f"{record['variant']}|{record['category']}|{record['key']}"
        if uid in seen:
            continue
        seen.add(uid)

        text = _build_chunk_text(record)
        metadata = {
            "variant": record["variant"],
            "category": record["category"],
            "key": record["key"],
            "key_en": record.get("key_en", ""),
            "value": record["value"],
            "note": record.get("note"),
            "source_url": record.get("source_url", source_url),
            "product_family": product_family,
        }
        chunks.append(Chunk(text=text, metadata=metadata))

    print(f"[chunker] Built {len(chunks)} chunks.")
    return chunks


def save_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    """將 chunks 存成 JSON，方便 debug。"""
    if path is None:
        path = PROCESSED_DIR / "chunks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [c.to_dict() for c in chunks]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[chunker] Saved {len(chunks)} chunks → {path}")
    return path


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """從磁碟載入 chunks。"""
    if path is None:
        path = PROCESSED_DIR / "chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(text=d["text"], metadata=d["metadata"]) for d in data]


if __name__ == "__main__":
    chunks = build_chunks()
    save_chunks(chunks)
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ---")
        print(c.text)
