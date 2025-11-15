"""
Upload KilterGPT model to Hugging Face Hub
Run this script from your project root directory
"""
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

# Configuration
HF_USERNAME = "nottreepat"  # Change this to your HF username
MODEL_NAME = "testtest" #"kilter-gpt"
REPO_ID = f"{HF_USERNAME}/{MODEL_NAME}"

# Path to your trained model directory
MODEL_DIR = "/Users/nottreepat/Downloads/checkpoint-78000"

def upload_model():
    """Upload model files to Hugging Face Hub"""
    
    # Create repository (if it doesn't exist)
    try:
        create_repo(REPO_ID, repo_type="model", exist_ok=True)
        print(f"✅ Repository created: {REPO_ID}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Initialize API
    api = HfApi()
    
    # Files to upload
    files_to_upload = [
        "pytorch_model.bin",
        "config.json",  # Make sure you have this
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    
    # Upload each file
    for filename in files_to_upload:
        file_path = Path(MODEL_DIR) / filename
        if file_path.exists():
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type="model"
            )
            print(f"✅ Uploaded {filename}")
        else:
            print(f"⚠️  {filename} not found at {file_path}")
    
    # Create a README.md
    readme_content = f"""---
language: en
tags:
- climbing
- route-generation
- gpt2
license: mit
---

# KilterGPT

A generative model for designing Kilter Board climbing routes.

## Model Description

KilterGPT generates climbing routes based on:
- Board angle
- Difficulty grade
- Required/starting holds

Built on GPT-2 architecture and trained on climbing route data.
"""
    
    readme_path = Path("README.md")
    readme_path.write_text(readme_content)
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model"
    )
    print(f"✅ Model uploaded successfully to: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload_model()