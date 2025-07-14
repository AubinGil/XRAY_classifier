# download_chestxray_subset.py

import os
import requests
import zipfile
import tarfile
import shutil

# === CONFIG ===
REPO_ZIP_URL     = "https://github.com/MichaelNoya/nih-chest-xray-webdataset-subset/archive/refs/heads/main.zip"
ZIP_FILENAME     = "subset.zip"
EXTRACT_DIR      = "repo_extract"
OUTPUT_DIR       = "chestxray_data"
IMAGES_DIR       = os.path.join(OUTPUT_DIR, "images")
TRAIN_LABELS_DIR = os.path.join(OUTPUT_DIR, "train_labels")
TEST_LABELS_DIR  = os.path.join(OUTPUT_DIR, "test_labels")
# ==============

def download_zip(url, dest):
    print(f"Downloading {url} …")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    print("Download complete.")

def extract_zip(zip_path, dest_dir):
    print(f"Extracting {zip_path} to {dest_dir} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    print("ZIP extraction complete.")

def extract_all_tars(src_root, images_out):
    print(f"Looking for .tar files under {src_root} …")
    for root, _, files in os.walk(src_root):
        for fn in files:
            if fn.endswith(".tar"):
                tar_path = os.path.join(root, fn)
                print(f"  → Extracting {fn}")
                with tarfile.open(tar_path, "r") as t:
                    t.extractall(images_out)
    print("All .tar shards extracted.")

def copy_labels(src_labels, dest_dir, filename):
    os.makedirs(dest_dir, exist_ok=True)
    src_file = os.path.join(src_labels, filename)
    if not os.path.isfile(src_file):
        raise FileNotFoundError(f"{filename} not found in {src_labels}")
    shutil.copy2(src_file, dest_dir)
    print(f"Copied {filename} → {dest_dir}")

def cleanup(zip_f, extract_dir):
    print("Cleaning up …")
    if os.path.exists(zip_f):
        os.remove(zip_f)
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)
    print("Cleanup done.")

if __name__ == "__main__":
    # 0) Prep output dirs
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    os.makedirs(TEST_LABELS_DIR, exist_ok=True)

    # 1) Download & unzip
    download_zip(REPO_ZIP_URL, ZIP_FILENAME)
    extract_zip(ZIP_FILENAME, EXTRACT_DIR)

    # 2) Locate repo root
    repo_root = next(
        os.path.join(EXTRACT_DIR, d)
        for d in os.listdir(EXTRACT_DIR)
        if os.path.isdir(os.path.join(EXTRACT_DIR, d))
    )

    # 3) Extract images
    extract_all_tars(os.path.join(repo_root, "datasets"), IMAGES_DIR)

    # 4) Copy train & test labels
    labels_src = os.path.join(repo_root, "labels")
    copy_labels(labels_src, TRAIN_LABELS_DIR, "train_labels.csv")
    copy_labels(labels_src, TEST_LABELS_DIR,  "test_labels.csv")

    # 5) Cleanup
    cleanup(ZIP_FILENAME, EXTRACT_DIR)

    # Summary
    total_imgs = sum(len(files) for _,_,files in os.walk(IMAGES_DIR))
    print(f"\n✅ Done! {total_imgs} images in {IMAGES_DIR}")
    print(f"✅ train_labels.csv in {TRAIN_LABELS_DIR}")
    print(f"✅ test_labels.csv  in {TEST_LABELS_DIR}")
