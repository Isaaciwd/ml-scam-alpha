#%%
import logging
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
import os
os.environ['TORCH_LOGS'] = '-fake_tensor'
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import sys
import time
import argparse
import random
import psutil # For system memory
import os # For process ID
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/glade/work/idavis/ml_scam/TCNN/')
sys.path.append('/glade/work/idavis/ml_scam/latent/')
from offline_torch_funcs_latent import train_model as train_offline_model, evaluate_loss as evaluate_offline_loss
from GlobalAtmoMixr import LatentGlobalAtmosMixer
from NNorm_GAM import NNorm_LatentGlobalAtmosMixer
from pathlib import Path
from config import tendency_variances_path as DEFAULT_TENDENCY_VARIANCES_PATH

# Helper function to print memory usage
def print_memory_usage(label: str):
    process = psutil.Process(os.getpid())
    rss_memory_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[MEM_DEBUG] {label}: RSS Memory = {rss_memory_gb:.3f} GB")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[MEM_DEBUG] {label}: CUDA Memory Allocated = {allocated_gb:.3f} GB, Reserved = {reserved_gb:.3f} GB")

# --- Set Global Random Seed ---
# This needs to be accessible by seed_worker, so define it globally or pass it appropriately.
# For simplicity here, we'll assume it's globally accessible for the worker_init_fn.
seed = 54 # Defined early for seed_worker
# print_memory_usage("Script Start") # Initial memory usage

# Top-level function for worker_init_fn
def seed_worker(worker_id):
    # worker_seed = torch.initial_seed() % 2**32 # Not needed if we use global seed
    # np.random.seed(worker_seed)
    # random.seed(worker_seed)
    # Instead, use the global seed + worker_id to ensure different seeds for workers
    # but deterministic behavior based on the main seed.
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    # It's generally not recommended to re-seed PyTorch's global RNG in worker_init_fn
    # if the main process already seeded it, unless specific per-worker seeding is intended
    # for torch operations within the worker itself (e.g., augmentations using torch.rand).
    # For numpy operations in data loading/preprocessing, seeding numpy is key.



#%%
class MultiComponentChunkedDataset(Dataset):
    def __init__(self, data_file_pattern: str):
        self.data_files = sorted(glob.glob(data_file_pattern))
        
        if not self.data_files:
            raise FileNotFoundError(f"No files found matching pattern: {data_file_pattern}")
            
        print(f"Found {len(self.data_files)} data chunk files.")

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx: int):
        file_path = self.data_files[idx]
        try:
            with open(file_path, 'rb') as f:
                data_chunk = torch.load(f, map_location='cpu', weights_only=False)
            # Extract needed tensors right away
            # inputs_tuple = (data_chunk_dict['profile_input'], 
            #                data_chunk_dict['forcing_input'])
            # targets_tuple = (data_chunk_dict['profile_target'],
            #                  )
            # Clear original dictionary reference
            # del data_chunk_dict
            return data_chunk
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise e

    
if __name__ == '__main__':
    # print_memory_usage("Inside __main__")
    # --- Set Random Seed ---
    # seed = 42 # Moved to be globally accessible for seed_worker
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # for multi-GPU.
    # You might also want to set these for full reproducibility, though they can impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # --- Model Architecture Selection ---
    model_architecture = "NNorm_LatentGlobalAtmosMixer"  # Options: "LatentGlobalAtmosMixer", "NNorm_LatentGlobalAtmosMixer"

    model_name = "NNormLatentGlobalAtmosMixer_small_MAE.pth"  # Name for the model checkpoint
    td_stem = "latent/window_3"
    print(f"Model architecture: {model_architecture}")
    print(f"Model name: {model_name}")
    print(f"td_stem: {td_stem}")
    # print_memory_usage("Before DataLoader initialization")

    # Parameters for training
    batch_size = 256 
    eval_steps = 5184 * 4 #2592 #20736  
    patience = 150 
    warmup_start_lr = 1e-5 # This is the initial learning rate for the warmup phase
    # This scalar determines how many evaluation cycles the warmup should span
    warmup_eval_cycles_scalar = 5 # howm many eval cycles to warmup over
    lr = 2e-4 # Changed from 1e-15 to a more practical fine-tuning LR
    epochs = 100
    num_workers = 5 
    prefetch_factor = 4 
    persistent_workers = True
    model_path = f'/glade/work/idavis/ml_scam/latent/train_encoder_mixer_decoder/models/{model_name}'
    load = True
    resume_training = False # If True, resumes full state (optimizer, scheduler, epoch, etc.)
    resume_optimizer_use_new_lr_config = True # New Flag: If True (and load=True, resume_training=False), loads optimizer state but uses script's LR/scheduler.
    reset_best_loss_on_load = True # If True and load=True, will not load best_test_loss from checkpoint
    train_data_stem  = f"/glade/derecho/scratch/idavis/ml_scam/data/held_suarez/{td_stem}"
    stats_path = Path("/glade/derecho/scratch/idavis/archive/held_suarez/atm/hist/stats/")
    use_mixed_precision = False
    weight_decay = 0.01 
    l1_target_ratio = 0.05
    cosine_loss_weight = 1.0 # Now acts as a multiplier on the dynamic scaling factor
    use_dynamic_cosine_scaling = False # Enable dynamic scaling by default
    print_interval_timing = True 
    use_torch_compile = True 
    grad_clip_max_norm = 1 # Default value for gradient clipping
    teacher_forcing_requires_grad = True  # Toggle gradient tracking on teacher-forcing branch
    grad_ckpt_level = 0  # 0: disabled, 1: block/per-head, 2: full forward checkpoint
    loss_component_weights = {
        'representation': 1.0,
        'true_decoder': 1.0,
        'full_state': 1.0,
        'true_tendency': 1.0,
        'pred_tendency': 2.0,
    }

    # --- Noise Injection Parameters ---
    noise_factor = 0 # Set to 0 to disable
    warmup_start_noise_factor = 0.0
    final_noise_factor = 0
    noise_std_dev_file = "/glade/work/idavis/ml_scam/TCNN/hybrid_cnn_transformer_v1_256_conv_correct_loss_no_cmpl_ckpt3_error_std_dev.nc"
    # These are the surface variables to which noise will be added.
    # The order must match the order in the input tensors.
    noise_surface_var_names = ['PS', 'SOLIN', 'SST', 'TREFHT'] #ORDER MATTERS! Alphabetical. Std devs of solin and SST are 0, so no noise will be added, but need them her

    # --- Physics/Regularization Parameters ---
    hydrostatic_lambda = 0.0 # Define hydrostatic_lambda
    lapse_rate_lambda = 0.0  # Define lapse_rate_lambda
    ds_pl_var_list = ["T","U","V","Z3"] # Define ds_pl_var_list for physics enforcements ORDER MATTERS! Alphabetical

    # --- Loss Function Configuration ---
    loss_type = 'MAE'  # Options: 'vicreg', 'barlow'
    barlow_lambda = 5e-3  # Lambda parameter for Barlow Twins loss (redundancy reduction weight)
    mse_loss_type = 'mae'  # Options: 'mse', 'mae'

    # --- Scheduler Configuration ---
    scheduler_type = 'LinearDecayLR' # Options: 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'LinearDecayLR'
    scheduler_patience = 5
    scheduler_factor = 0.6
    T_0 = eval_steps * 50 # Used by CosineAnnealingWarmRestarts (number of batches for the first restart)
    T_mult = 2           # Used by CosineAnnealingWarmRestarts
    eta_min_cosine = 1e-8 # Explicit eta_min for CosineAnnealingLR or CosineAnnealingWarmRestarts
    # For CosineAnnealingLR, if scheduler_type is 'CosineAnnealingLR':
    # T_max will be set to cosine_lr_t_max_batches, and stepped per batch.
    # Example: if 100 epochs, 1000 batches/epoch, then 100000 for one full cycle over training.
    cosine_lr_t_max_batches = eval_steps * 23  # Set T_max in batches for CosineAnnealingLR. Stepped per batch.
    linear_decay_total_batches = eval_steps * 20  # Example: total batches over all epochs for LinearDecayLR
    linear_decay_end_lr = 1e-5  # Changed from 1e-10 to ensure decay from new lr

    # --- Print Full Training Configuration ---
    print("--- Full Training Configuration ---")
    print(f"Model Architecture: {model_architecture}")
    print(f"Model Name: {model_name}")
    print(f"Training Data Stem: {train_data_stem}")
    print(f"Stats Path: {stats_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Weight Decay: {weight_decay}")
    print(f"L1 Target Ratio: {l1_target_ratio}")
    print(f"Cosine Loss Weight (Multiplier): {cosine_loss_weight}")
    print(f"Use Dynamic Cosine Scaling: {use_dynamic_cosine_scaling}")
    print(f"Gradient Clipping Max Norm: {grad_clip_max_norm}")
    print(f"Use Mixed Precision: {use_mixed_precision}")
    print(f"Use torch.compile: {use_torch_compile}")
    print(f"Load Model: {load}, Resume Training: {resume_training}")
    print(f"Teacher Forcing Requires Grad: {teacher_forcing_requires_grad}")
    print(f"Gradient Checkpoint Level: {grad_ckpt_level}")
    if load:
        print(f"  Reset best loss on load: {reset_best_loss_on_load}")
    print(f"Scheduler: {scheduler_type}")
    if scheduler_type == 'ReduceLROnPlateau':
        print(f"  Scheduler Patience: {scheduler_patience}")
        print(f"  Scheduler Factor: {scheduler_factor}")
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        print(f"  T_0: {T_0}")
        print(f"  T_mult: {T_mult}")
        print(f"  eta_min: {eta_min_cosine}")
    elif scheduler_type == 'CosineAnnealingLR':
        print(f"  T_max (batches): {cosine_lr_t_max_batches}")
        print(f"  eta_min: {eta_min_cosine}")
    elif scheduler_type == 'LinearDecayLR':
        print(f"  Total Decay Batches: {linear_decay_total_batches}")
        print(f"  End LR: {linear_decay_end_lr}")
    print(f"Warmup Eval Cycles Scalar: {warmup_eval_cycles_scalar}")
    print(f"Warmup Start LR: {warmup_start_lr}")
    print(f"Noise Factor: {noise_factor}")
    if noise_factor > 0:
        print(f"  Warmup Start Noise Factor: {warmup_start_noise_factor}")
        print(f"  Final Noise Factor: {final_noise_factor}")
        print(f"  Noise Std Dev File: {noise_std_dev_file}")
    print(f"Hydrostatic Lambda: {hydrostatic_lambda}")
    print(f"Lapse Rate Lambda: {lapse_rate_lambda}")
    print(f"Loss Function: {loss_type}")
    if loss_type == 'barlow':
        print(f"  Barlow Lambda: {barlow_lambda}")
    print(f"MSE Loss Type: {mse_loss_type}")
    print(f"Loss Component Weights: {loss_component_weights}")
    print("-------------------------------------")

    # files_train = f"{train_data_stem}/data_dict_000*.pt"
    # files_test = f"{train_data_stem}/test/data_dict*pt"
    files_train = f"{train_data_stem}/data_dict_*.pt"
    files_test = f"{train_data_stem}/test/data_dict_*pt"
    # print("WARNING: ONLY USING 26 LEVELS OF DATA")
    # print()
    # print("WARNING: ONLY USING 26 LEVELS OF DATA")
    # print()
    # print("WARNING: ONLY USING 26 LEVELS OF DATA")
    # print()
    # print("WARNING: ONLY USING 26 LEVELS OF DATA")
    # --- Model Configuration Parameters ---
    supported_architectures = {
        "LatentGlobalAtmosMixer": {
            "class": LatentGlobalAtmosMixer,
            "config": {
                'M_features': 122,
                'model_dim': 256,
                'n_heads': 16,
                'num_mixer_blocks': 2,
                'head_mlp_hidden_dims': [128],
                'ffn_hidden_dims': [1024],
                'encoder_hidden_dims': [192],
                'dropout_rate': 0.1,
                'teacher_forcing_requires_grad': teacher_forcing_requires_grad,
                'grad_ckpt_level': grad_ckpt_level,
            },
        },

        "NNorm_LatentGlobalAtmosMixer": {
            "class": NNorm_LatentGlobalAtmosMixer,
            "config": {
                'M_features': 122,
                'model_dim': 256,
                'n_heads': 16,
                'num_mixer_blocks': 2,
                'head_mlp_hidden_dims': [128],
                'ffn_hidden_dims': [1024],
                'encoder_hidden_dims': [192],
                'dropout_rate': 0.1,
                'teacher_forcing_requires_grad': teacher_forcing_requires_grad,
                'grad_ckpt_level': grad_ckpt_level,
            },
        },
    }

    if model_architecture not in supported_architectures:
        raise ValueError(f"Unsupported model_architecture '{model_architecture}'. Options: {list(supported_architectures.keys())}")

    model_entry = supported_architectures[model_architecture]
    model_class = model_entry["class"]
    base_model_config = model_entry["config"].copy()
    model_config_params = base_model_config.copy()

    print(f"l1 target ratio: {l1_target_ratio}")
    print(f"Using mixed precision: {use_mixed_precision}")
    print(f"Using scheduler: {scheduler_type}")
    if load:
        print(f"Load model: True, Resume full training state: {resume_training}")
        if not resume_training and resume_optimizer_use_new_lr_config:
            print(f"Resume optimizer state with new LR/scheduler: {resume_optimizer_use_new_lr_config}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # print("WARNING: Using smalWARNINGl subset of test data for testing. Ensure this is appropriate for your use case.")

    if epochs == 1:
        print("Warning: epochs = 1. Ensure test set is appropriately defined or not used for final model saving if it's part of training data.")

    train_dataset = MultiComponentChunkedDataset(files_train)
    test_dataset = MultiComponentChunkedDataset(files_test)

    if len(train_dataset) == 0:
        print(f"No training files found with pattern: {files_train}. Exiting.")
        sys.exit(1)
    if len(test_dataset) == 0:
        print(f"No test files found with pattern: {files_test}. Exiting.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker # Use the top-level function
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, 
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker # Use the top-level function
    )

    print("Finished DataLoader init")
    print(f"Train dataset size (number of chunk files): {len(train_dataset)}")
    print(f"Test dataset size (number of chunk files): {len(test_dataset)}")
    # print_memory_usage("After DataLoader initialization")
    
    global model
    loaded_checkpoint = None
    active_model_config = model_config_params.copy() # Default to script's config

    if load:
        print(f"Attempting to load model from: {model_path}")
        # print_memory_usage(f"Before torch.load for {model_path}")
        try:
            loaded_checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            # print_memory_usage(f"After torch.load for {model_path}")
            
            # Use model config from checkpoint if available
            loaded_model_config_from_checkpoint = loaded_checkpoint['hyperparams'].get('model_config_params')
            if loaded_model_config_from_checkpoint:
                print("Loaded model_config_params from checkpoint. Using these for model instantiation.")
                active_model_config = loaded_model_config_from_checkpoint.copy()
            else:
                print("Warning: model_config_params not found in checkpoint hyperparams. Using current script's config for model instantiation.")
                # active_model_config remains the script's model_config_params

            ckpt_mse_loss_type = loaded_checkpoint['hyperparams'].get('mse_loss_type') if loaded_checkpoint.get('hyperparams') else None
            if ckpt_mse_loss_type and ckpt_mse_loss_type != mse_loss_type:
                print(f"Warning: Checkpoint was trained with mse_loss_type='{ckpt_mse_loss_type}'. Overriding script value ('{mse_loss_type}') to match checkpoint.")
                mse_loss_type = ckpt_mse_loss_type

            if model_architecture == "LatentGlobalAtmosMixer":
                active_model_config['teacher_forcing_requires_grad'] = teacher_forcing_requires_grad
                active_model_config['grad_ckpt_level'] = grad_ckpt_level
            
            saved_architecture = loaded_checkpoint['hyperparams'].get('architecture')

            # Architecture compatibility check
            if saved_architecture == "OptimizedModule":
                print(f"Warning: Checkpoint architecture is {saved_architecture}. "
                      f"The script is configured to use {model_class.__name__}. "
                      f"Proceeding with loading, assuming {saved_architecture} is a compiled version of {model_class.__name__} "
                      f"and that the loaded model_config_params are compatible.")
            elif saved_architecture and saved_architecture != model_class.__name__:
                print(f"Error: Checkpoint architecture ({saved_architecture}) "
                      f"does not match current model_class ({model_class.__name__}). Exiting.")
                sys.exit(1)
            elif not saved_architecture:
                 print(f"Warning: 'architecture' not found in checkpoint hyperparams. "
                       f"Proceeding with current model_class {model_class.__name__}, assuming compatibility.")


            model = model_class(**active_model_config) # Instantiate with the chosen config and selected class
            
            # Prepare state_dict for loading
            state_dict_to_load = loaded_checkpoint['model_state_dict']
            if saved_architecture == "OptimizedModule":
                # If the saved model was compiled (e.g. "OptimizedModule"), 
                # its state_dict keys might be prefixed (e.g. "_orig_mod.").
                # We are loading into a fresh, uncompiled instance of model_class, so strip the prefix if present.
                if any(k.startswith("_orig_mod.") for k in state_dict_to_load.keys()):
                    print("Adjusting state_dict keys from OptimizedModule (stripping '_orig_mod.' prefix).")
                    new_state_dict = {
                        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                        for k, v in state_dict_to_load.items()
                    }
                    state_dict_to_load = new_state_dict
                else:
                    print("OptimizedModule state_dict does not appear to contain '_orig_mod.' prefix. Loading keys as is.")

            model.load_state_dict(state_dict_to_load)
            print("Model state loaded successfully.")
            # print_memory_usage("After model.load_state_dict")

        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Training from scratch.")
            # print_memory_usage("After FileNotFoundError during load")
            load = False
            loaded_checkpoint = None
            resume_training = False 
            active_model_config = model_config_params.copy() # Reset to script's config if loading failed
        except Exception as e:
            print(f"Error loading model checkpoint: {e}. Training from scratch.")
            load = False
            loaded_checkpoint = None
            resume_training = False
            active_model_config = model_config_params.copy() # Reset to script's config if loading failed

    if model_architecture == "LatentGlobalAtmosMixer":
        active_model_config['teacher_forcing_requires_grad'] = teacher_forcing_requires_grad
        active_model_config['grad_ckpt_level'] = grad_ckpt_level

    if not load: # This executes if load was initially False OR if loading failed
        print(f"Initializing new {model_class.__name__} model...")
        # print_memory_usage("Before new model initialization")
        # active_model_config is already model_config_params if load is false or failed and not updated from checkpoint
        model = model_class(**active_model_config) 
        # print_memory_usage("After new model initialization")
        resume_training = False 
        loaded_checkpoint = None

    model_config_params = active_model_config.copy()
    if hasattr(model, "set_teacher_forcing_grad"):
        model.set_teacher_forcing_grad(teacher_forcing_requires_grad)
    if hasattr(model, "set_grad_ckpt_level"):
        model.set_grad_ckpt_level(grad_ckpt_level)

    model.to(device)
    print(f"Using model: {type(model).__name__}")
    print(f"Model configuration used for instantiation: {active_model_config}")
    print(f"Saving to: {model_path}")
    # print_memory_usage("Before train_model call")

    # if load and loaded_checkpoint: # Only evaluate if a model was actually loaded
    #     print("Evaluating loss of loaded model on test set before resuming/starting training...")
    #     # The fourth argument to evaluate_loss is eval_batch_size, which must be an integer.
    #     # l1_target_ratio was incorrectly passed here. Using the script's batch_size.
    #     initial_eval_batch_size = batch_size*16 # Or consider batch_size * 16 if consistent with other eval calls
    #     initial_test_loss = evaluate_loss(model, test_loader, device, initial_eval_batch_size, use_mixed_precision=use_mixed_precision)
    #     print(f"Initial test loss of loaded model: {initial_test_loss}")

    torch.set_float32_matmul_precision('high')

    warmup_parameter_for_train_model = warmup_eval_cycles_scalar * eval_steps

    # max_files_to_evaluate = 256
    # avg_test_loss, test_data_loading_time = evaluate_loss(model, test_loader, device, 2048*16, use_mixed_precision, max_files_to_evaluate= max_files_to_evaluate)
    # print(f"  Test Loss:  {avg_test_loss:.6f}, Data Loading Time: {test_data_loading_time:.3f} sec")


    # --- Training ---
    print(f"Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    train_losses, test_losses, lr_history, noise_factor_history, model = train_offline_model(
        model, train_loader, test_loader, device, model_config_params,
        epochs=epochs, lr=lr, batch_size=batch_size, eval_steps=eval_steps,
        patience=patience, model_path=model_path,
        l1_target_ratio=l1_target_ratio,
        warmup_eval_steps=warmup_parameter_for_train_model,
        warmup_start_lr=warmup_start_lr,
        scheduler_type=scheduler_type, scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor, T_0=T_0, T_mult=T_mult,
        eta_min_cosine=eta_min_cosine,
        cosine_lr_t_max_batches=cosine_lr_t_max_batches,
        linear_decay_total_batches=linear_decay_total_batches,
        linear_decay_end_lr=linear_decay_end_lr,
        use_mixed_precision=use_mixed_precision,
        loaded_checkpoint=loaded_checkpoint if load else None,
        resume_training=resume_training,
        resume_optimizer_use_new_lr_config=resume_optimizer_use_new_lr_config,
        reset_best_loss_on_load=reset_best_loss_on_load,
        weight_decay=weight_decay,
        hydrostatic_lambda=hydrostatic_lambda,
        lapse_rate_lambda=lapse_rate_lambda,
        ds_pl_var_list=ds_pl_var_list,
        print_interval_timing=print_interval_timing,
        use_torch_compile=use_torch_compile,
        grad_clip_max_norm=grad_clip_max_norm,
        noise_factor=noise_factor,
        warmup_start_noise_factor=warmup_start_noise_factor,
        final_noise_factor=final_noise_factor,
        noise_std_dev_file=noise_std_dev_file,
        noise_surface_var_names=noise_surface_var_names,
        cosine_loss_weight=cosine_loss_weight,
        use_dynamic_cosine_scaling=use_dynamic_cosine_scaling,
        loss_type=loss_type,
        barlow_lambda=barlow_lambda,
        loss_weights=loss_component_weights,
        mse_loss_type=mse_loss_type,
        tendency_variances_path=DEFAULT_TENDENCY_VARIANCES_PATH,
    )
    # print_memory_usage("After train_model call")

    # Final evaluation after training completion
    if load and loaded_checkpoint:
        print("Evaluating loss of loaded model on test set after training...")
        final_eval_batch_size = batch_size * 16
        tendency_variances = torch.load(Path(DEFAULT_TENDENCY_VARIANCES_PATH)).to(device)
        tend_var_mean = tendency_variances.mean()
        tendency_variances = torch.cat([tendency_variances, tend_var_mean.view(1), tend_var_mean.view(1)], dim=0)
        if mse_loss_type == 'mae':
            tendency_variances = torch.sqrt(tendency_variances)
        avg_test_loss, test_data_loading_time = evaluate_offline_loss(
            model, test_loader, device, final_eval_batch_size, 
            use_mixed_precision=use_mixed_precision,
            loss_type=loss_type,
            barlow_lambda=barlow_lambda,
            tendency_variances=tendency_variances,
            loss_weights=loss_component_weights,
            mse_loss_type=mse_loss_type,
        )
        print(f"Final test loss of loaded model: {avg_test_loss:.6f}, Data Loading Time: {test_data_loading_time:.3f} sec")
