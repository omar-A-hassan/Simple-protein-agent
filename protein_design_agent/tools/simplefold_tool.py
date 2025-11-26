import os
import sys
import logging
import uuid
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SimpleFold configuration
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

    try:
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

            if _PRECOMPUTED_EMBEDDINGS is not None:
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
        return _fold_with_simplefold(sequence, job_id, output_dir)
    except Exception as e:
        logger.error(f"Error during folding: {str(e)}")
        return {
            "status": "error",
            "error_message": f"Folding failed: {str(e)}"
        }


def _fold_with_simplefold(sequence: str, job_id: str, output_dir: str) -> Dict[str, Any]:
    """
    SimpleFold implementation using Apple's ml-simplefold with sequential ESM-3B loading.

    Uses a two-stage approach to avoid OOM on limited memory environments:
    1. Load ESM-3B, compute embeddings, unload
    2. Run SimpleFold with pre-computed embeddings
    """
    logger.info("Running SimpleFold with sequential loading...")

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
        # NOTE: SimpleFold's run_inference expects plddt_model to be a dict, not None
        # When plddt is disabled, we pass a dict with plddt_out_module set to None
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
        else:
            # Create dummy plddt_model dict with expected structure but None values
            # SimpleFold expects multiple keys in plddt_model dict when plddt is disabled
            plddt_model = {
                "plddt_out_module": None,
                "plddt_latent_module": None,
                "plddt_embed_module": None,
                "plddt_linear_module": None
            }

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

        if results is None:
            raise RuntimeError("SimpleFold inference returned None")

        # Save results
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
