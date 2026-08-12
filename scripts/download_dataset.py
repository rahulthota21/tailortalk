"""
Downloads the saree image dataset (from the Google Drive link given in the
assignment PDF) and unzips it into data/raw/.

Run this once, locally, before build_index.py:
    python scripts/download_dataset.py
"""
import os
import zipfile
import gdown

DRIVE_URL = "https://drive.google.com/file/d/1EsXFleb1RhU7ylU76WpBwSwcgDw7R8tV/view?usp=drive_link"
OUT_ZIP = "data/dataset.zip"
OUT_DIR = "data/raw"

os.makedirs("data", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("Downloading dataset from Google Drive...")
gdown.download(url=DRIVE_URL, output=OUT_ZIP, quiet=False, fuzzy=True)

print("Unzipping...")
with zipfile.ZipFile(OUT_ZIP, "r") as zf:
    zf.extractall(OUT_DIR)

print(f"Done. Images extracted to {OUT_DIR}/")
print("If the drive link is actually a folder rather than a single zip, "
      "download it manually from the link and place the images under "
      f"{OUT_DIR}/ instead.")
