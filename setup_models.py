import os
import sys
import subprocess
from pathlib import Path

# Configuration
ARTIFACTS_DIR = Path("artifacts")
MODELS = {
    "simplefold_100M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt",
    "plddt_module": "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt",
    "simplefold_1.6B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.6B.ckpt",
    "esm_3b": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt",
    "esm_3b_regression": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt"
}

def download_file(url, dest_path):
    """Download a file using wget with resume support and progress bar."""
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping.")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
        # Use wget for robust downloading
        subprocess.run(
            ["wget", "-c", "-O", str(dest_path), url],
            check=True
        )
        print(f"Successfully downloaded {dest_path}")
    except subprocess.CalledProcessError:
        print(f"Failed to download {url}. Retrying with curl...")
        try:
            subprocess.run(
                ["curl", "-L", "-o", str(dest_path), url],
                check=True
            )
            print(f"Successfully downloaded {dest_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {url}: {e}")
            sys.exit(1)

def main():
    # Create artifacts directory
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting SimpleFold Model Setup...")
    
    # 1. Download the base folding model (100M)
    download_file(MODELS["simplefold_100M"], ARTIFACTS_DIR / "simplefold_100M.ckpt")
    
    # 2. Download pLDDT dependencies (Optional but requested)
    # NOTE: pLDDT requires the 1.6B model which causes OOM in standard Colab.
    # We are disabling this by default to save 6GB+ of download.
    # print("\n📦 Downloading pLDDT (Confidence) Models...")
    # print("NOTE: This includes a large 1.6B model (~6GB). Please be patient.")
    
    # download_file(MODELS["plddt_module"], ARTIFACTS_DIR / "plddt_module_1.6B.ckpt")
    # download_file(MODELS["simplefold_1.6B"], ARTIFACTS_DIR / "simplefold_1.6B.ckpt")
    
    print("\n✅ Setup Complete! Base model (100M) is ready.")
    
    # 3. Download ESM-3B Model (Required for SimpleFold)
    print("\n📦 Downloading ESM-3B Model (Required)...")
    # We download to the torch hub cache directory to match where the tool looks for it
    torch_hub_dir = Path.home() / ".cache/torch/hub/checkpoints"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)
    
    download_file(MODELS["esm_3b"], torch_hub_dir / "esm2_t36_3B_UR50D.pt")
    download_file(MODELS["esm_3b_regression"], torch_hub_dir / "esm2_t36_3B_UR50D-contact-regression.pt")
    
    print("\n✅ ESM-3B Model Downloaded.")

if __name__ == "__main__":
    main()
