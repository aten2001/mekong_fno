import os
from huggingface_hub import HfApi

def main():
    repo_id = os.environ.get("HF_DATASET_REPO", "").strip()
    token   = os.environ.get("HF_TOKEN", "").strip()
    out_dir = os.environ.get("OUT_DIR", "out")

    if not repo_id:
        raise RuntimeError("HF_DATASET_REPO is missing")
    if not token:
        raise RuntimeError("HF_TOKEN is missing")

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=out_dir,
        path_in_repo="",
        commit_message="update live_backfill + status (automated)",
        token=token,
    )
    print("[upload] ok ->", repo_id, "from", out_dir)

if __name__ == "__main__":
    main()