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
    """Download a file using wget with resume support, fallback to curl if wget fails."""
    if dest_path.exists():
        print(f"File {dest_path} already exists. Skipping.")
        return

    print(f"Downloading {url} to {dest_path}...")
    try:
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
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting SimpleFold Model Setup...")

    # Download SimpleFold 100M model
    download_file(MODELS["simplefold_100M"], ARTIFACTS_DIR / "simplefold_100M.ckpt")

    # pLDDT models are disabled by default to avoid OOM on limited memory environments
    # Uncomment below to enable pLDDT confidence prediction (requires ~6GB additional download)
    # download_file(MODELS["plddt_module"], ARTIFACTS_DIR / "plddt_module_1.6B.ckpt")
    # download_file(MODELS["simplefold_1.6B"], ARTIFACTS_DIR / "simplefold_1.6B.ckpt")

    print("\nSetup Complete! Base model (100M) is ready.")

    # Download ESM-3B model to torch hub cache
    print("\nDownloading ESM-3B Model (Required)...")
    torch_hub_dir = Path.home() / ".cache/torch/hub/checkpoints"
    torch_hub_dir.mkdir(parents=True, exist_ok=True)

    download_file(MODELS["esm_3b"], torch_hub_dir / "esm2_t36_3B_UR50D.pt")
    download_file(MODELS["esm_3b_regression"], torch_hub_dir / "esm2_t36_3B_UR50D-contact-regression.pt")

    print("\nESM-3B Model Downloaded.")

if __name__ == "__main__":
    main()
