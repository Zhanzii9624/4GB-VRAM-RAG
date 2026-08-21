"""
rag/parser.py
定義 GIGABYTE AORUS MASTER 16 AM6H 規格資料，輸出成 specs.json
資料為手動整理自官網(https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp)
採手動維護並定期人工核對的方式確保準確性

schema:
{
  "product_family": str,
  "source_url": str,
  "fetched_at": str,
  "records": [
    {"variant": "shared"|"BZH"|"BYH"|"BXH",
     "category": str,
     "key": str,
     "key_en": str,
     "value": str,
     "note": str|None}
  ]
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_URL = "https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def _record(
    category: str,
    key: str,
    key_en: str,
    value: str,
    variant: str = "shared",
    note: str | None = None,
) -> dict:
    return {
        "variant": variant,
        "category": category,
        "key": key,
        "key_en": key_en,
        "value": value,
        "note": note,
    }


SPECS: dict[str, Any] = {
    "product_family": "AORUS MASTER 16 AM6H",
    "source_url": SOURCE_URL,
    "fetched_at": "2026-08-22T00:00:00+00:00",
    "records": [
        _record("作業系統", "作業系統", "Operating System",
                "Windows 11 Pro / Windows 11 Home / UEFI Shell OS"),
        _record("處理器", "處理器", "CPU / Processor",
                "Intel Core Ultra 9 Processor 275HX (36MB cache, up to 5.4GHz, 24 cores, 24 threads)"),
        _record("顯示器", "顯示器", "Display / Screen",
                "16-inch 16:10 OLED WQXGA (2560x1600), 240Hz, 1ms, DCI-P3 100%, "
                "500nits peak, 1,000,000:1 contrast; NVIDIA G-SYNC; NVIDIA Advanced Optimus; "
                "VESA DisplayHDR True Black 500; VESA ClearMR 10000; Pantone Validated; "
                "TÜV Rheinland Low Blue Light; Dolby Vision"),
        _record("記憶體", "記憶體", "Memory / RAM",
                "Up to 64GB DDR5 5600MHz (2 x SO-DIMM slots for expansion)"),
        _record("儲存裝置", "儲存裝置", "Storage",
                "1x PCIe Gen5 M.2 slot + 1x PCIe Gen4x4 M.2 slot, up to 4TB PCIe NVMe M.2 SSD",
                note="儲存容量依國家地區出貨而異"),
        _record("鍵盤", "鍵盤", "Keyboard",
                "3-zone RGB Backlit Keyboard, up to 1.7mm key-travel (support N-Key rollover)"),
        _record("連接埠", "連接埠", "Ports / I/O",
                "Left: 1x DC-in, 1x RJ-45, 1x HDMI 2.1, 1x USB3.2 Gen2 Type-A, "
                "1x Thunderbolt 5 Type-C (USB4, DisplayPort 2.1, Power Delivery 3.0). "
                "Right: 1x USB3.2 Gen2 Type-A, 1x Thunderbolt 4 Type-C (USB4, DisplayPort 1.4, Power Delivery 3.0), "
                "1x MicroSD (UHS-II), 1x Audio Jack (mic/headphone combo)"),
        _record("音效", "音效", "Audio / Sound",
                "4x 2W speakers, Microphone, Dolby Atmos, Smart Amp Technology"),
        _record("通訊", "有線網路", "LAN", "LAN: 1G"),
        _record("通訊", "無線網路", "Wi-Fi / Wireless", "WIFI 7 (802.11be 2x2)"),
        _record("通訊", "藍牙", "Bluetooth", "Bluetooth v5.4"),
        _record("視訊鏡頭", "視訊鏡頭", "Webcam / Camera",
                "FHD (1080p) IR Webcam, built-in array microphone, supports Windows Hello Face Authentication"),
        _record("安全裝置", "安全性", "Security",
                "Firmware-based TPM, supports Intel Platform Trust Technology (Intel PTT)"),
        _record("電池", "電池", "Battery", "Li-ion 99Wh"),
        _record("變壓器", "變壓器", "Power Adapter / AC Adapter", "330W AC Adapter"),
        _record("尺寸", "尺寸", "Dimensions / Size", "357 x 254 x 23~29.9 mm",
                note="實際尺寸依配置、製程與量測方式而異"),
        _record("重量", "重量", "Weight", "約 2.5 kg",
                note="實際重量依配置、製程與量測方式而異"),
        _record("顏色", "顏色", "Color", "Dark Tide"),

        # GPU 依 variant 不同
        _record("顯示晶片", "顯示晶片", "GPU / Graphics",
                "NVIDIA GeForce RTX 5090 Laptop GPU, 24GB GDDR7, 175W Maximum Graphics Power with Dynamic Boost",
                variant="BZH"),
        _record("顯示晶片", "顯示晶片", "GPU / Graphics",
                "NVIDIA GeForce RTX 5080 Laptop GPU, 16GB GDDR7, 175W Maximum Graphics Power with Dynamic Boost",
                variant="BYH"),
        _record("顯示晶片", "顯示晶片", "GPU / Graphics",
                "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12GB GDDR7, 140W Maximum Graphics Power with Dynamic Boost",
                variant="BXH"),
    ],
}


def build_specs() -> dict[str, Any]:
    """輸出 specs.json，並回傳 specs dict。"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "specs.json"
    out_path.write_text(json.dumps(SPECS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(SPECS['records'])} records -> {out_path}")
    return SPECS


if __name__ == "__main__":
    build_specs()