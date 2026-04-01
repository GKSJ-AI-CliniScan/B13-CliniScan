import os
import sys
from huggingface_hub import HfApi, login

def deploy_to_hf(repo_id="mediscan-api"):
    try:
        api = HfApi()
        # Try to get user identity to see if logged in
        user = api.whoami()
        username = user['name']
        full_repo_id = f"{username}/{repo_id}"
        print(f"Logged in as {username}. Deploying to {full_repo_id}...")
        
        # Create Space if it doesn't exist
        try:
            api.create_repo(repo_id=full_repo_id, repo_type="space", space_sdk="docker", exist_ok=False)
            print(f"Created new Space: {full_repo_id}")
        except Exception as e:
            if "already exists" in str(e).lower() or "409" in str(e):
                print(f"Space {full_repo_id} already exists. Updating...")
            else:
                raise e
        
        # Upload directory
        print("Uploading files... This may take a while depending on model sizes.")
        api.upload_folder(
            folder_path=".",
            repo_id=full_repo_id,
            repo_type="space",
            ignore_patterns=[
                ".git/*", "venv/*", "env/*", "__pycache__/*", "*.pyc",
                ".DS_Store", "data/*", "data_prep/*", "docs/*", "results/*"
            ]
        )
        print(f"Deployment successful! Your API should soon be live at:")
        print(f"https://{username}-{repo_id}.hf.space")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you are logged into Hugging Face.")
        print("Run 'huggingface-cli login' in your terminal and provide a token with WRITE access.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        deploy_to_hf(sys.argv[1])
    else:
        deploy_to_hf()
