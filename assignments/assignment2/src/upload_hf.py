"""
Step 5: Upload detection Parquet to HuggingFace Hub.
Set HF_REPO_ID env var and login first: huggingface-cli login
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = BASE_DIR / "outputs" / "detections.parquet"
REPO_ID = os.environ.get("HF_REPO_ID", "ps5761/car-parts-retrieval")


def upload():
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"{PARQUET_PATH} not found. Run detect.py first.")

    api = HfApi()
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(PARQUET_PATH),
        path_in_repo="detections.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"[done] Uploaded to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    upload()