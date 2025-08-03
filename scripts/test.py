import requests
from huggingface_hub import HfApi

api = HfApi()
repo = "BilltheTsar/shuttle-detection-dataset"
siblings = api.dataset_info(repo).siblings

total_size = 0
print("Fetching file sizes (using HEAD requests)...")
for sibling in siblings:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{sibling.rfilename}"
    response = requests.head(url, allow_redirects=True)
    size = int(response.headers.get("content-length", 0))
    total_size += size
    print(f"{sibling.rfilename}: {size / (1024**2):.2f} MB")

print(f"\nEstimated total size: {total_size / (1024**3):.2f} GB")
