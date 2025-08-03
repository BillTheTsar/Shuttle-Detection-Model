import shutil
import zipfile
from huggingface_hub import hf_hub_download
import os

# Hugging Face repo info
REPO_ID = "BillTheTsar/shuttle-detection-zips"
ZIP_FILES = [f"{i:03}.zip" for i in range(1, 12)]  # 001.zip to 011.zip
EXTRACT_DIR = "/storage/dataset"

os.makedirs(EXTRACT_DIR, exist_ok=True)

for zip_name in ZIP_FILES:
    print(f"Downloading {zip_name}...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=zip_name,
        repo_type="dataset"
    )

    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Clean cache after each file to save space
    print("Clearing cache...")
    shutil.rmtree("/root/.cache/huggingface", ignore_errors=True)

print("✅ All zip files downloaded and extracted!")