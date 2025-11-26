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
    "plddt": False,                    # Disabled: 1.6B pLDDT model causes OOM crash in Colab
    "simplefold_repo_path": None       # Path to ml-simplefold repo (set via env or auto-detect)
}

# Global storage for pre-computed embeddings
_PRECOMPUTED_EMBEDDINGS = None

def _generate_embeddings_sequentially(sequence: str):
    """
    Generates ESM-3B embeddings sequentially:
    1. Load ESM-3B model
    2. Compute embeddings
    3. Unload model and clear memory
    """
    global _PRECOMPUTED_EMBEDDINGS
    global _PRECOMPUTED_EMBEDDINGS
    logger.warning("DEBUG: Starting sequential embedding generation...")
    
    try:
        import torch
        import torch
        import esm
        import esm.pretrained

        # PATCH: Fix for missing ablation models in esm.pretrained
        # These functions are missing in some versions of fair-esm but required by ml-simplefold
        missing_models = [
            "esmfold_structure_module_only_8M",
            "esmfold_structure_module_only_8M_270K",
            "esmfold_structure_module_only_35M",
            "esmfold_structure_module_only_35M_270K",
            "esmfold_structure_module_only_150M",
            "esmfold_structure_module_only_150M_270K",
            "esmfold_structure_module_only_650M",
            "esmfold_structure_module_only_650M_270K",
            "esmfold_structure_module_only_3B",
            "esmfold_structure_module_only_3B_270K",
            "esmfold_structure_module_only_15B",
            "esmfold_structure_module_only_15B_270K",
        ]
        
        for model_name in missing_models:
            if not hasattr(esm.pretrained, model_name):
                logger.warning(f"Patching missing {model_name} in esm.pretrained")
                # Create a dummy function that raises NotImplementedError
                # We need to capture model_name in the closure
                def _create_dummy(name):
                    def _dummy():
                        raise NotImplementedError(f"This is a dummy function for {name} patched by Protein Agent.")
                    return _dummy
                
                setattr(esm.pretrained, model_name, _create_dummy(model_name))

        # Load ESM-3B model
        logger.info("Loading ESM-3B model (esm2_t36_3B_UR50D)...")
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("ESM-3B loaded on GPU")
        else:
            logger.info("ESM-3B loaded on CPU")

        # Prepare input
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_len = batch_tokens.shape[1]
        
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.cuda()

        # Generate embeddings from all 37 layers (required by SimpleFold)
        logger.info("Computing embeddings from all 37 layers...")
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=list(range(37)), return_contacts=True)

        # Stack all layers: shape will be [batch, seq_len, 37, embed_dim]
        all_layers = [results["representations"][i] for i in range(37)]
        token_representations = torch.stack(all_layers, dim=2)

        # Move to CPU and store
        _PRECOMPUTED_EMBEDDINGS = token_representations.cpu()
        logger.info(f"Embeddings computed. Shape: {_PRECOMPUTED_EMBEDDINGS.shape}")
        
        # CLEANUP
        del results
        del token_representations
        del model
        del alphabet
        del batch_converter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ESM-3B unloaded and memory cleared.")
        
        return True

    except Exception as e:
        logger.error(f"Failed to generate embeddings sequentially: {e}")
        return False

def _apply_sequential_patch():
    """
    Monkey-patches esm_utils to use pre-computed embeddings.
    """
    try:
        # PATCH: Fix for missing ablation models in esm.pretrained
        # We must apply this BEFORE importing esm_utils
        import esm.pretrained
        missing_models = [
            "esmfold_structure_module_only_8M",
            "esmfold_structure_module_only_8M_270K",
            "esmfold_structure_module_only_35M",
            "esmfold_structure_module_only_35M_270K",
            "esmfold_structure_module_only_150M",
            "esmfold_structure_module_only_150M_270K",
            "esmfold_structure_module_only_650M",
            "esmfold_structure_module_only_650M_270K",
            "esmfold_structure_module_only_3B",
            "esmfold_structure_module_only_3B_270K",
            "esmfold_structure_module_only_15B",
            "esmfold_structure_module_only_15B_270K",
        ]
        
        for model_name in missing_models:
            if not hasattr(esm.pretrained, model_name):
                logger.warning(f"Patching missing {model_name} in esm.pretrained (Global Patch)")
                def _create_dummy(name):
                    def _dummy():
                        raise NotImplementedError(f"This is a dummy function for {name} patched by Protein Agent.")
                    return _dummy
                setattr(esm.pretrained, model_name, _create_dummy(model_name))

        from src.simplefold.utils import esm_utils
        
        # Store original if needed (though we mostly bypass it)
        if not hasattr(esm_utils, "_original_compute"):
            esm_utils._original_compute = esm_utils.compute_language_model_representations

        def patched_compute(model, alphabet, batch_tokens):
            """
            Replacement function that returns pre-computed embeddings.
            Ignores the passed model (which might be None or dummy).
            """
            global _PRECOMPUTED_EMBEDDINGS
            logger.warning("DEBUG: Accessed patched compute_language_model_representations")
            logger.warning(f"DEBUG: batch_tokens shape: {batch_tokens.shape if hasattr(batch_tokens, 'shape') else 'N/A'}")

            if _PRECOMPUTED_EMBEDDINGS is not None:
                logger.warning(f"DEBUG: Returning pre-computed embeddings with shape: {_PRECOMPUTED_EMBEDDINGS.shape}")
                logger.warning(f"DEBUG: Embeddings dtype: {_PRECOMPUTED_EMBEDDINGS.dtype}, device: {_PRECOMPUTED_EMBEDDINGS.device}")
                return _PRECOMPUTED_EMBEDDINGS
            else:
                logger.error("No pre-computed embeddings found!")
                raise RuntimeError("Sequential loading failed: Embeddings not found.")

        esm_utils.compute_language_model_representations = patched_compute
        logger.info("Sequential patch applied to esm_utils")
        return True
        
    except ImportError:
        logger.error("Could not import esm_utils to patch")
        return False

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

    # Apply SEQUENTIAL patch
    logger.info("Applying sequential loading patch...")
    patch_success = _apply_sequential_patch()
    if not patch_success:
        logger.error("Sequential patch FAILED")
    else:
        logger.info("Sequential patch applied.")

    # 1. Generate Embeddings Sequentially
    logger.info("STEP 1: Generating embeddings with ESM-3B...")
    if not _generate_embeddings_sequentially(sequence):
        raise RuntimeError("Failed to generate embeddings.")
    
    # 2. Proceed with SimpleFold (which will use the patched function)


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
        logger.warning("DEBUG: Processing sequence and running inference...")
        batch, structure, record = inference_wrapper.process_input(sequence)
        logger.warning(f"DEBUG: process_input done. Batch type: {type(batch)}")

        logger.warning("DEBUG: About to call run_inference...")
        try:
            results = inference_wrapper.run_inference(batch, folding_model, plddt_model, device=device)
            logger.warning(f"DEBUG: run_inference completed successfully. Results type: {type(results)}")
        except Exception as e:
            logger.error(f"CRITICAL: run_inference raised an exception!")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception message: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

        # Check what's actually in results before accessing it
        if results is None:
            logger.error("CRITICAL: run_inference returned None!")
            raise RuntimeError("run_inference returned None - check SimpleFold implementation")
        else:
            logger.warning(f"DEBUG: Results is not None. Type: {type(results)}")
            if hasattr(results, 'keys'):
                logger.warning(f"DEBUG: Results keys: {list(results.keys())}")
            elif isinstance(results, (list, tuple)):
                logger.warning(f"DEBUG: Results is a {type(results)} with length {len(results)}")
            else:
                logger.warning(f"DEBUG: Results is {type(results)}: {results}")

        # Save results (returns list of paths)
        logger.warning("DEBUG: About to call save_result...")
        save_paths = inference_wrapper.save_result(structure, record, results, out_name=job_id)
        logger.warning(f"DEBUG: save_result done. save_paths: {save_paths}")
        
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
