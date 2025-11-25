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
    "num_steps": 50,                   # REDUCED from 500 to 50 to prevent Colab crashes
    "tau": 0.05,                       # Temperature parameter
    "nsample_per_protein": 1,          # Number of structures to generate
    "plddt": False,                    # Disabled by default: requires downloading huge 1.6B model which fails in Colab
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

    # SimpleFold requires BOTH paths:
    # 1. repo_root: to import 'src.simplefold.wrapper'
    # 2. src/simplefold: to allow internal imports like 'from model.flow import...'
    
    if repo_path is None:
        raise RuntimeError(
            "SimpleFold repository not found. Please set SIMPLEFOLD_REPO_PATH environment variable "
            "or clone ml-simplefold to the current directory."
        )
    
    # Add repo root
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    # Add src/simplefold for internal module resolution
    inner_path = str(Path(repo_path) / "src" / "simplefold")
    if inner_path not in sys.path:
        sys.path.insert(0, inner_path)
    
    try:
        # Import exactly as shown in SimpleFold's sample.ipynb
        from src.simplefold.wrapper import ModelWrapper, InferenceWrapper
        import lightning.pytorch as pl
        pl.seed_everything(42, workers=True)
    except ImportError as e:
        raise RuntimeError(
            f"Failed to import SimpleFold modules. Ensure ml-simplefold is installed: {e}"
        )

    # SimpleFold uses relative paths for configs, so we MUST run from the repo directory
    original_cwd = os.getcwd()
    os.chdir(repo_path)
    logger.info(f"Changed working directory to {repo_path} for SimpleFold execution")

    try:
        # Initialize ModelWrapper
        logger.info(f"Loading SimpleFold model: {SIMPLEFOLD_CONFIG['model_size']}")
        model_wrapper = ModelWrapper(
            simplefold_model=SIMPLEFOLD_CONFIG["model_size"],
            ckpt_dir=SIMPLEFOLD_CONFIG["ckpt_dir"],
            plddt=SIMPLEFOLD_CONFIG["plddt"],
            backend=SIMPLEFOLD_CONFIG["backend"]
        )

        device = model_wrapper.device
        
        # Helper to load model with auto-recovery for corrupted checkpoints
        def load_model_safe(load_func, model_name_for_path):
            try:
                return load_func()
            except RuntimeError as e:
                # Check for zip corruption error (common if download was interrupted)
                if "failed finding central directory" in str(e) or "PytorchStreamReader" in str(e):
                    logger.warning(f"Detected corrupted checkpoint for {model_name_for_path}. Deleting and retrying...")
                    
                    # Construct path to checkpoint (relative to current CWD which is repo_path)
                    # Note: SimpleFold names files as {model_name}.ckpt
                    ckpt_filename = f"{model_name_for_path}.ckpt"
                    ckpt_path = os.path.join(SIMPLEFOLD_CONFIG["ckpt_dir"], ckpt_filename)
                    
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)
                        logger.info(f"Deleted corrupted checkpoint: {ckpt_path}")
                    else:
                        logger.warning(f"Could not find checkpoint file at {ckpt_path} to delete.")
                    
                    # Retry loading (SimpleFold wrapper will re-download)
                    logger.info("Retrying model load (this will trigger re-download)...")
                    return load_func()
                else:
                    raise e

        # Load folding model
        folding_model = load_model_safe(
            model_wrapper.from_pretrained_folding_model, 
            SIMPLEFOLD_CONFIG["model_size"]
        )
        
        # Load pLDDT model if requested
        plddt_model = None
        if SIMPLEFOLD_CONFIG["plddt"]:
            # pLDDT model has a specific name in SimpleFold wrapper, usually plddt_module_1.6B.ckpt
            # But the wrapper handles the name internally. We just need to catch the error.
            # For the file path, we might guess, but let's just try/except the call.
            # The wrapper uses a fixed URL/name for pLDDT: plddt_module_1.6B.ckpt
            
            try:
                plddt_model = model_wrapper.from_pretrained_plddt_model()
            except RuntimeError as e:
                 if "failed finding central directory" in str(e) or "PytorchStreamReader" in str(e):
                    logger.warning("Detected corrupted pLDDT checkpoint. Deleting and retrying...")
                    ckpt_path = os.path.join(SIMPLEFOLD_CONFIG["ckpt_dir"], "plddt_module_1.6B.ckpt")
                    if os.path.exists(ckpt_path):
                        os.remove(ckpt_path)
                    plddt_model = model_wrapper.from_pretrained_plddt_model()
                 else:
                    raise e

        # Initialize InferenceWrapper
        # Note: output_dir is relative to the NEW cwd (repo_path)
        # We want the output to be in the original output_dir, so we need absolute path
        abs_output_dir = os.path.abspath(os.path.join(original_cwd, output_dir))
        prediction_dir = os.path.join(abs_output_dir, "simplefold_predictions")
        
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
        
        # Return absolute path
        return {
            "status": "success",
            "pdb_file": str(Path(output_pdb_path).absolute()),
            "job_id": job_id
        }

    finally:
        # Always restore original CWD
        os.chdir(original_cwd)
        logger.info(f"Restored working directory to {original_cwd}")


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
