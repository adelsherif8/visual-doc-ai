"""One-shot deploy of this app to a Hugging Face Space (Gradio SDK).

Usage:
    pip install huggingface_hub
    huggingface-cli login            # paste your HF *write* token (stays local)
    python deploy_hf.py              # creates/updates the Space and prints the URL

Then on the Space page: Settings -> Variables and secrets -> add
    OPENAI_API_KEY = <your NEW key>          (required for live uploads)
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (optional, for traces)
The Space rebuilds and your public link goes live. With no secret set it still
runs in offline MOCK mode over the bundled samples. (packages.txt installs
tesseract-ocr on the Space for OCR on uploaded documents.)

Never commit your API key — it belongs in the Space secrets, not the repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, whoami
except ImportError:
    sys.exit("Missing dependency. Run:  pip install huggingface_hub")

REPO_NAME = "visual-doc-ai"
HERE = Path(__file__).resolve().parent
IGNORE = [
    ".venv/*", "**/__pycache__/*", "*.pyc", ".git/*",
    ".gradio/*", ".env", ".DS_Store",
]


def main() -> None:
    try:
        user = whoami()["name"]
    except Exception:
        sys.exit("Not logged in. Run:  huggingface-cli login")

    repo_id = f"{user}/{REPO_NAME}"
    api = HfApi()

    print(f"→ Creating/locating Space {repo_id} …")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    print("→ Uploading files (this can take a minute) …")
    api.upload_folder(
        folder_path=str(HERE),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy Visual Document AI",
    )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print("\n✅ Done.")
    print(f"   Space:  {url}")
    print(f"   App:    https://{user.lower()}-{REPO_NAME}.hf.space")
    print("\nNext: open the Space → Settings → add OPENAI_API_KEY as a secret,")
    print("then wait for it to rebuild. (It runs in MOCK mode until you do.)")


if __name__ == "__main__":
    main()
