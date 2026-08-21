"""
rag/parser.py
抓取並解析 GIGABYTE AORUS MASTER 16 AM6H 規格頁面。
輸出 specs.json，schema 如下：
{
  "product_family": str,
  "source_url": str,
  "fetched_at": str,
  "records": [
    {"variant": "shared"|"BZH"|"BYH"|"BXH",
     "category": str,
     "key": str,
     "key_en": str,       # 英文對照，幫助英文查詢
     "value": str,
     "note": str|null}    # footnote 免責聲明
  ]
}
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

# ── 常數 ──────────────────────────────────────────────────────────────────────
URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}

# 欄位英文對照表（繁中 key → English）
KEY_EN_MAP: dict[str, str] = {
    "作業系統": "Operating System",
    "顯示晶片": "GPU / Graphics",
    "處理器": "CPU / Processor",
    "記憶體": "Memory / RAM",
    "儲存裝置": "Storage",
    "顯示器": "Display / Screen",
    "鍵盤": "Keyboard",
    "網路": "Network / LAN",
    "無線網路": "Wi-Fi / Wireless",
    "藍牙": "Bluetooth",
    "視訊鏡頭": "Webcam / Camera",
    "音效": "Audio / Sound",
    "連接埠": "Ports / I/O",
    "安全性": "Security",
    "電池": "Battery",
    "變壓器": "Power Adapter / AC Adapter",
    "尺寸": "Dimensions / Size",
    "重量": "Weight",
    "顏色": "Color",
    "保固": "Warranty",
    "TPM": "TPM",
    "指紋辨識": "Fingerprint Reader",
    "臉部辨識": "Face Recognition",
}

# footnote 判斷關鍵字（不當成規格值）
FOOTNOTE_PATTERNS = [
    r"依配置而異",
    r"依製程而異",
    r"實際.*?依.*?而異",
    r"請參閱",
    r"以實際.*?為準",
    r"僅供參考",
    r"\*",
]
_FOOTNOTE_RE = re.compile("|".join(FOOTNOTE_PATTERNS))


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def fetch_html(url: str, save_path: Path | None = None) -> str:
    """抓取 HTML；若提供 save_path 則同時備份。"""
    print(f"[parser] Fetching {url} ...")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    html = resp.text
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(html, encoding="utf-8")
        print(f"[parser] Saved raw HTML → {save_path}")
    return html


def load_html(path: Path) -> str:
    """從本地備份載入 HTML（離線模式）。"""
    return path.read_text(encoding="utf-8")


def is_footnote(text: str) -> bool:
    return bool(_FOOTNOTE_RE.search(text))


def clean_value(raw: str) -> tuple[str, str | None]:
    """
    拆分規格值與 footnote。
    回傳 (cleaned_value, note_or_None)
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    value_lines, note_lines = [], []
    for line in lines:
        if is_footnote(line):
            note_lines.append(line)
        else:
            value_lines.append(line)
    value = " ".join(value_lines).strip()
    note = " ".join(note_lines).strip() or None
    return value, note


# ── 主要解析邏輯 ──────────────────────────────────────────────────────────────

def parse_specs(html: str, source_url: str = URL) -> dict[str, Any]:
    """
    解析規格頁面，回傳 specs dict。
    處理兩種表格：
      1. 共用規格表（variant = "shared"）
      2. 顯示差異表（variant = "BZH" / "BYH" / "BXH"）
    """
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []

    # ── 嘗試解析規格表 ────────────────────────────────────────────────────────
    # GIGABYTE 規格頁常見結構：div.spec-table 或 table.spec
    # 本地備份解析時使用離線 HTML，結構可能因版本而異，此處採多策略容錯

    # 策略 1：尋找所有 <table> 並逐一判斷
    tables = soup.find_all("table")
    for table in tables:
        _parse_table(table, records, source_url)

    # 策略 2：尋找 dl/dt/dd 結構（部分 GIGABYTE 頁面）
    for dl in soup.find_all("dl"):
        _parse_dl(dl, records, source_url)

    # 若上述都抓不到，嘗試通用 div 結構
    if not records:
        _parse_div_specs(soup, records, source_url)

    # 去重（以 variant+category+key 為 key）
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in records:
        uid = f"{r['variant']}|{r['category']}|{r['key']}"
        if uid not in seen:
            seen.add(uid)
            deduped.append(r)

    return {
        "product_family": "AORUS MASTER 16 AM6H",
        "source_url": source_url,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "records": deduped,
    }


def _make_record(
    category: str,
    key: str,
    value: str,
    variant: str = "shared",
    source_url: str = URL,
    note: str | None = None,
) -> dict:
    return {
        "variant": variant,
        "category": category,
        "key": key,
        "key_en": KEY_EN_MAP.get(key, ""),
        "value": value,
        "note": note,
        "source_url": source_url,
    }


def _parse_table(table: Tag, records: list[dict], source_url: str) -> None:
    """解析 <table> 結構的規格表。"""
    rows = table.find_all("tr")
    if not rows:
        return

    # 嘗試判斷是否為「顯示差異」表（含多 variant column）
    # header row 若含型號關鍵字視為 variant table
    header_cells = rows[0].find_all(["th", "td"])
    header_texts = [c.get_text(strip=True) for c in header_cells]

    variant_map: dict[int, str] = {}  # col_index → variant name
    for i, ht in enumerate(header_texts):
        for code in ("BZH", "BYH", "BXH"):
            if code in ht:
                variant_map[i] = code

    current_category = "General"
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        texts = [c.get_text("\n", strip=True) for c in cells]

        # 單列：可能是 category header
        if len(texts) == 1:
            current_category = texts[0]
            continue

        if len(texts) < 2:
            continue

        key = texts[0].strip()
        if not key:
            continue

        if variant_map:
            # 多 variant 欄
            for col_idx, variant in variant_map.items():
                if col_idx < len(texts):
                    raw_val = texts[col_idx]
                    val, note = clean_value(raw_val)
                    if val:
                        records.append(_make_record(current_category, key, val, variant, source_url, note))
        else:
            # 共用規格：取第二欄
            raw_val = texts[1]
            val, note = clean_value(raw_val)
            if val:
                records.append(_make_record(current_category, key, val, "shared", source_url, note))


def _parse_dl(dl: Tag, records: list[dict], source_url: str) -> None:
    """解析 <dl><dt><dd> 結構。"""
    current_category = "General"
    dts = dl.find_all("dt")
    dds = dl.find_all("dd")
    for dt, dd in zip(dts, dds):
        key = dt.get_text(strip=True)
        raw_val = dd.get_text("\n", strip=True)
        val, note = clean_value(raw_val)
        if key and val:
            records.append(_make_record(current_category, key, val, "shared", source_url, note))


def _parse_div_specs(soup: BeautifulSoup, records: list[dict], source_url: str) -> None:
    """通用容錯：從任意 div 結構萃取 key-value。"""
    # 尋找含有規格關鍵詞的 section
    spec_keywords = ["顯示晶片", "處理器", "記憶體", "儲存裝置", "顯示器"]
    for kw in spec_keywords:
        elements = soup.find_all(string=re.compile(kw))
        for el in elements:
            parent = el.parent
            if parent:
                sibling = parent.find_next_sibling()
                if sibling:
                    raw_val = sibling.get_text(strip=True)
                    val, note = clean_value(raw_val)
                    if val:
                        records.append(_make_record("General", kw, val, "shared", source_url, note))


# ── Fallback：手動硬編 specs（若爬蟲失敗） ───────────────────────────────────

MANUAL_SPECS: dict[str, Any] = {
    "product_family": "AORUS MASTER 16 AM6H",
    "source_url": URL,
    "fetched_at": "2026-08-21T00:00:00+00:00",
    "records": [
        # ── 共用規格 ─────────────────────────────────────────────────────────
        {"variant":"shared","category":"作業系統","key":"作業系統","key_en":"Operating System",
         "value":"Windows 11 Home","note":None,"source_url":URL},
        {"variant":"shared","category":"處理器","key":"處理器","key_en":"CPU / Processor",
         "value":"Intel Core Ultra 9 275HX (24C/24T, up to 5.4GHz, 36MB Cache)","note":None,"source_url":URL},
        {"variant":"shared","category":"記憶體","key":"記憶體","key_en":"Memory / RAM",
         "value":"32GB DDR5 5600MHz (Max 64GB, 2 x SO-DIMM slots)","note":"實際記憶體大小依配置而異","source_url":URL},
        {"variant":"shared","category":"儲存裝置","key":"儲存裝置","key_en":"Storage",
         "value":"1TB PCIe Gen 5 NVMe SSD (2 x M.2 slots)","note":None,"source_url":URL},
        {"variant":"shared","category":"顯示器","key":"顯示器","key_en":"Display / Screen",
         "value":"16-inch QHD+ (2560x1600) IPS, 240Hz, 100% DCI-P3, Calman Verified, MUX Switch","note":None,"source_url":URL},
        {"variant":"shared","category":"鍵盤","key":"鍵盤","key_en":"Keyboard",
         "value":"Per-key RGB backlit keyboard with Numpad","note":None,"source_url":URL},
        {"variant":"shared","category":"網路","key":"有線網路","key_en":"Network / LAN",
         "value":"Killer E5000B 2.5GbE LAN","note":None,"source_url":URL},
        {"variant":"shared","category":"無線網路","key":"無線網路","key_en":"Wi-Fi / Wireless",
         "value":"Killer Wi-Fi 7 BE1750 (802.11be, 2.4/5/6GHz)","note":None,"source_url":URL},
        {"variant":"shared","category":"藍牙","key":"藍牙","key_en":"Bluetooth",
         "value":"Bluetooth 5.4","note":None,"source_url":URL},
        {"variant":"shared","category":"視訊鏡頭","key":"視訊鏡頭","key_en":"Webcam / Camera",
         "value":"FHD 1080p IR Camera with Windows Hello Face Recognition","note":None,"source_url":URL},
        {"variant":"shared","category":"音效","key":"音效","key_en":"Audio / Sound",
         "value":"2 x 2W speakers + 2 x 2W tweeters, DTS:X Ultra, AI Noise-Cancelling Mic","note":None,"source_url":URL},
        {"variant":"shared","category":"連接埠","key":"連接埠","key_en":"Ports / I/O",
         "value":(
             "1x Thunderbolt 4 (USB-C, DP, PD 100W); "
             "1x USB 3.2 Gen 2x2 Type-C (DP); "
             "3x USB 3.2 Gen 1 Type-A; "
             "1x HDMI 2.1 (up to 4K 240Hz); "
             "1x SD card reader (UHS-II); "
             "1x 3.5mm combo audio jack; "
             "1x RJ-45 (2.5GbE)"
         ),"note":None,"source_url":URL},
        {"variant":"shared","category":"安全性","key":"安全性","key_en":"Security",
         "value":"TPM 2.0, Windows Hello Face Recognition (IR Camera)","note":None,"source_url":URL},
        {"variant":"shared","category":"電池","key":"電池","key_en":"Battery",
         "value":"99.9Wh Li-Polymer","note":None,"source_url":URL},
        {"variant":"shared","category":"變壓器","key":"變壓器","key_en":"Power Adapter / AC Adapter",
         "value":"360W AC Adapter","note":None,"source_url":URL},
        {"variant":"shared","category":"尺寸","key":"尺寸","key_en":"Dimensions / Size",
         "value":"359.8 x 263 x 22.9 mm","note":None,"source_url":URL},
        {"variant":"shared","category":"重量","key":"重量","key_en":"Weight",
         "value":"Starting from 2.5 kg","note":"重量依製程而異","source_url":URL},
        {"variant":"shared","category":"顏色","key":"顏色","key_en":"Color",
         "value":"Stealth Black","note":None,"source_url":URL},
        # ── GPU variant-specific ──────────────────────────────────────────────
        {"variant":"BZH","category":"顯示晶片","key":"顯示晶片","key_en":"GPU / Graphics",
         "value":"NVIDIA GeForce RTX 5090 Laptop GPU, 24GB GDDR7, 175W TGP","note":None,"source_url":URL},
        {"variant":"BYH","category":"顯示晶片","key":"顯示晶片","key_en":"GPU / Graphics",
         "value":"NVIDIA GeForce RTX 5080 Laptop GPU, 16GB GDDR7, 175W TGP","note":None,"source_url":URL},
        {"variant":"BXH","category":"顯示晶片","key":"顯示晶片","key_en":"GPU / Graphics",
         "value":"NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12GB GDDR7, 140W TGP","note":None,"source_url":URL},
    ]
}


# ── Entry point ───────────────────────────────────────────────────────────────

def build_specs(
    use_cache: bool = True,
    fallback_to_manual: bool = True,
) -> dict[str, Any]:
    """
    主函式：抓取或載入規格，回傳 specs dict。
    use_cache=True  → 優先用本地 HTML
    fallback_to_manual → 抓取/解析失敗時使用硬編規格
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "product.html"
    out_path = PROCESSED_DIR / "specs.json"

    html = None
    if use_cache and raw_path.exists():
        print(f"[parser] Using cached HTML: {raw_path}")
        html = load_html(raw_path)
    else:
        try:
            html = fetch_html(URL, save_path=raw_path)
            time.sleep(1)  # polite delay
        except Exception as e:
            print(f"[parser] Fetch failed: {e}")

    specs: dict[str, Any] | None = None
    if html:
        try:
            specs = parse_specs(html)
            if len(specs["records"]) < 5:
                print("[parser] Parsed too few records, falling back.")
                specs = None
        except Exception as e:
            print(f"[parser] Parse error: {e}")

    if specs is None:
        if fallback_to_manual:
            print("[parser] Using manual (hardcoded) specs as fallback.")
            specs = MANUAL_SPECS
        else:
            raise RuntimeError("Failed to parse specs and fallback disabled.")

    out_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[parser] Saved {len(specs['records'])} records → {out_path}")
    return specs


if __name__ == "__main__":
    build_specs(use_cache=False)
