import os
import sys
import logging
import uuid
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration: Set to True to use real SimpleFold, False for mock mode
USE_REAL_SIMPLEFOLD = os.environ.get("USE_REAL_SIMPLEFOLD", "false").lower() == "true"

# SimpleFold configuration (only used if USE_REAL_SIMPLEFOLD=True)
SIMPLEFOLD_CONFIG = {
    "model_size": "simplefold_100M",  # Options: 100M, 360M, 700M, 1.1B, 1.6B, 3B
    "ckpt_dir": "artifacts",           # Where SimpleFold model checkpoints are stored
    "backend": "mlx",                  # "mlx" for Apple Silicon, "torch" for others
    "num_steps": 500,                  # Inference steps
    "tau": 0.05,                       # Temperature parameter
    "nsample_per_protein": 1,          # Number of structures to generate
    "plddt": True,                     # Include confidence scores
    "simplefold_repo_path": None       # Path to ml-simplefold repo (set via env or auto-detect)
}

def fold_sequence(sequence: str) -> Dict[str, Any]:
    """
    Predicts the 3D structure of a protein sequence using Apple's SimpleFold.

    Args:
        sequence: The amino acid sequence of the protein (e.g., "MVLSPADKT...").

    Returns:
        A dictionary containing the path to the generated PDB file and status.
    """
    logger.info(f"Received request to fold sequence: {sequence[:10]}... (Length: {len(sequence)})")

    # Validate sequence
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    if not all(aa.upper() in valid_amino_acids for aa in sequence):
        return {
            "status": "error",
            "error_message": "Invalid amino acid sequence. Please use standard one-letter codes."
        }

    # Generate a unique ID for this job
    job_id = str(uuid.uuid4())
    output_dir = "output_pdbs"
    os.makedirs(output_dir, exist_ok=True)

    try:
        if USE_REAL_SIMPLEFOLD:
            return _fold_with_simplefold(sequence, job_id, output_dir)
        else:
            return _fold_mock(sequence, job_id, output_dir)

    except Exception as e:
        logger.error(f"Error during folding: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Folding failed: {str(e)}"
        }


def _fold_with_simplefold(sequence: str, job_id: str, output_dir: str) -> Dict[str, Any]:
    """
    Real SimpleFold implementation using Apple's ml-simplefold.

    Installation:
    1. Clone the repo: git clone https://github.com/apple/ml-simplefold.git
    2. Install: cd ml-simplefold && pip install -e .
    3. For Apple Silicon: pip install mlx==0.28.0
    4. Set environment variable: export USE_REAL_SIMPLEFOLD=true
    5. Set SIMPLEFOLD_REPO_PATH if needed
    """
    logger.info("Running with REAL SimpleFold model...")

    # Add SimpleFold to Python path
    repo_path = SIMPLEFOLD_CONFIG["simplefold_repo_path"]
    if repo_path is None:
        # Try to auto-detect in common locations
        possible_paths = [
            Path("./ml-simplefold"),
            Path("../ml-simplefold"),
            Path.home() / "ml-simplefold"
        ]
        for p in possible_paths:
            if p.exists():
                repo_path = str(p)
                break

    # SimpleFold uses src layout - add src directory to path
    if repo_path is None:
        raise RuntimeError(
            "SimpleFold repository not found. Please set SIMPLEFOLD_REPO_PATH environment variable "
            "or clone ml-simplefold to the current directory."
        )
    
    # Add the src directory to Python path for src layout
    src_path = str(Path(repo_path) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    try:
        from simplefold.wrapper import ModelWrapper, InferenceWrapper
        import lightning.pytorch as pl
        pl.seed_everything(42, workers=True)
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import SimpleFold modules. Ensure ml-simplefold is installed: {e}"
        )

    # Initialize ModelWrapper
    logger.info(f"Loading SimpleFold model: {SIMPLEFOLD_CONFIG['model_size']}")
    model_wrapper = ModelWrapper(
        simplefold_model=SIMPLEFOLD_CONFIG["model_size"],
        ckpt_dir=SIMPLEFOLD_CONFIG["ckpt_dir"],
        plddt=SIMPLEFOLD_CONFIG["plddt"],
        backend=SIMPLEFOLD_CONFIG["backend"]
    )

    device = model_wrapper.device
    folding_model = model_wrapper.from_pretrained_folding_model()
    plddt_model = model_wrapper.from_pretrained_plddt_model() if SIMPLEFOLD_CONFIG["plddt"] else None

    # Initialize InferenceWrapper
    prediction_dir = os.path.join(output_dir, "simplefold_predictions")
    inference_wrapper = InferenceWrapper(
        output_dir=SIMPLEFOLD_CONFIG["ckpt_dir"],
        prediction_dir=prediction_dir,
        num_steps=SIMPLEFOLD_CONFIG["num_steps"],
        tau=SIMPLEFOLD_CONFIG["tau"],
        nsample_per_protein=SIMPLEFOLD_CONFIG["nsample_per_protein"],
        device=device,
        backend=SIMPLEFOLD_CONFIG["backend"]
    )

    # Process input and run inference
    logger.info("Processing sequence and running inference...")
    batch, structure, record = inference_wrapper.process_input(sequence)
    results = inference_wrapper.run_inference(batch, folding_model, plddt_model, device=device)

    # Save results (returns list of paths)
    save_paths = inference_wrapper.save_result(structure, record, results, out_name=job_id)
    output_pdb_path = save_paths[0] if save_paths else None

    if output_pdb_path is None:
        raise RuntimeError("SimpleFold did not generate output file")

    logger.info(f"SimpleFold prediction saved to {output_pdb_path}")

    return {
        "status": "success",
        "pdb_file_path": os.path.abspath(output_pdb_path),
        "message": f"Successfully folded sequence of length {len(sequence)} using SimpleFold.",
        "job_id": job_id,
        "model_used": SIMPLEFOLD_CONFIG["model_size"],
        "confidence_included": SIMPLEFOLD_CONFIG["plddt"]
    }


def _fold_mock(sequence: str, job_id: str, output_dir: str) -> Dict[str, Any]:
    """
    Mock implementation for testing without SimpleFold installed.
    Creates a minimal valid PDB file.
    """
    logger.info("Running in MOCK mode. Simulating SimpleFold inference...")

    output_pdb_path = os.path.join(output_dir, f"{job_id}.pdb")

    # Create a dummy PDB file
    with open(output_pdb_path, "w") as f:
        f.write(f"REMARK   1 CREATED BY PROTEIN DESIGN AGENT (MOCK SIMPLEFOLD)\n")
        f.write(f"REMARK   2 SEQUENCE: {sequence}\n")
        f.write(f"REMARK   3 This is a mock PDB file for testing purposes\n")
        f.write(f"REMARK   4 To use real SimpleFold, set USE_REAL_SIMPLEFOLD=true\n")
        # A minimal valid PDB line for visualization tools not to crash immediately
        f.write("ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  MET A   1      11.000  10.000  10.000  1.00  0.00           C\n")
        f.write("END\n")

    logger.info(f"Mock PDB saved to {output_pdb_path}")

    return {
        "status": "success",
        "pdb_file_path": os.path.abspath(output_pdb_path),
        "message": f"Successfully folded sequence of length {len(sequence)} (MOCK MODE).",
        "job_id": job_id,
        "note": "This is a mock result. Set USE_REAL_SIMPLEFOLD=true to use real SimpleFold."
    }
