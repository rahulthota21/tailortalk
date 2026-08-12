"""
Walks data/raw/, embeds every image (DINOv2 + colour histogram), and writes
them into the local Chroma vector store at chroma_db/. Also looks up each
image's original public image_url from the CSV, so the deployed app can
display matches without needing the raw image files to be present on the
server.

Run once after download_from_csv.py:
    python scripts/build_index.py

Safe to re-run: already-indexed image ids are skipped, so it only embeds
new images if you add more to data/raw/.
"""
import os
import sys
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_store import get_collection, add_image  # noqa: E402

RAW_DIR = "data/raw"
CSV_PATH = "data/byrappa_tejas_31july.csv"
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def load_sku_to_url():
    mapping = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sku = row.get("SKU", "").strip()
            url = row.get("image_url", "").strip()
            if sku and url:
                mapping[sku] = url
    return mapping


def find_images(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(VALID_EXT):
                yield os.path.join(dirpath, fn)


def main():
    collection = get_collection()
    existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
    sku_to_url = load_sku_to_url()
    paths = list(find_images(RAW_DIR))
    print(f"Found {len(paths)} images in {RAW_DIR}/")

    added = 0
    for i, path in enumerate(paths):
        image_id = os.path.relpath(path, RAW_DIR)
        if image_id in existing_ids:
            continue

        sku = os.path.splitext(os.path.basename(path))[0]
        image_url = sku_to_url.get(sku, "")

        try:
            add_image(collection, image_id, path, image_url=image_url)
            added += 1
        except Exception as e:
            print(f"  skipping {path}: {e}")

        if (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(paths)}")

    print(f"Added {added} new images. Total items in collection: {collection.count()}")


if __name__ == "__main__":
    main()