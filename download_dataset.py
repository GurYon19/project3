import sys
import subprocess

def install_and_import():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download
    return snapshot_download

def main():
    snapshot_download = install_and_import()
    print("Starting download...")
    snapshot_download(
        repo_id="HumynLabs/Street-videos", 
        repo_type="dataset", 
        local_dir="data/Street-videos",
        local_dir_use_symlinks=False
    )
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
