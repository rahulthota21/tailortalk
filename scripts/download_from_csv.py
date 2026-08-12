"""
Downloads every saree image referenced in the byrappa CSV into data/raw/.
Run: python scripts/download_from_csv.py
"""
import os
import csv
import requests

CSV_PATH = "data/byrappa_tejas_31july.csv"
OUT_DIR = "data/raw"
os.makedirs(OUT_DIR, exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Found {len(rows)} rows.")

ok, failed = 0, 0
for i, row in enumerate(rows):
    sku = row.get("SKU", "").strip()
    url = row.get("image_url", "").strip()
    if not sku or not url:
        continue

    ext = os.path.splitext(url)[1] or ".webp"
    out_path = os.path.join(OUT_DIR, f"{sku}{ext}")
    if os.path.exists(out_path):
        ok += 1
        continue

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with open(out_path, "wb") as img_f:
            img_f.write(resp.content)
        ok += 1
    except Exception as e:
        failed += 1
        print(f"  failed {sku}: {e}")

    if (i + 1) % 50 == 0:
        print(f"  processed {i + 1}/{len(rows)}  (ok={ok}, failed={failed})")

print(f"Done. ok={ok}, failed={failed}. Images saved to {OUT_DIR}/")