import logging
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
import os
os.environ['TORCH_LOGS'] = '-fake_tensor'
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import time
import numpy as np
import xarray as xr
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, Dataset
import gc
import tqdm
import torch.nn.init as init
import copy
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import math # Added for cosine warmup
import psutil # For system memory
import os # For process ID
from pathlib import Path
from typing import Dict, Optional
# from lightly.loss import VICRegLoss
from VicRegLoss import VICRegLoss
from BarlowTwinsLoss import BarlowTwinsLoss


DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    'representation': 1.0,
    'true_decoder': 1.0,
    'full_state': 1.0,
    'true_tendency': 1.0,
    'pred_tendency': 2.0,
}


# Helper function to print memory usage
# def print_memory_usage(label: str):
def print_memory_usage(label: str):
    process = psutil.Process(os.getpid())
    rss_memory_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[MEM_DEBUG] {label}: RSS Memory = {rss_memory_gb:.3f} GB")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[MEM_DEBUG] {label}: CUDA Memory Allocated = {allocated_gb:.3f} GB, Reserved = {reserved_gb:.3f} GB")

def reshape_for_mixer(profile_input, forcing_input, profile_target):
    
    batch_size, n_input_vars, n_levels, window_dim, _ = profile_input.shape
    _, n_forcing_vars, _, _ = forcing_input.shape
    _, n_output_vars, _ = profile_target.shape

    profile_input = profile_input.permute(0, 3, 4, 1, 2).reshape(batch_size, window_dim , window_dim, n_input_vars * n_levels)

    _, n_forcing_vars, _, _ = forcing_input.shape
    forcing_input = forcing_input.permute(0, 2,3,1).reshape(batch_size, window_dim , window_dim, n_forcing_vars)
    input_tensor = torch.cat([profile_input, forcing_input], dim=3)

    target_tensor = profile_target.reshape(batch_size, n_output_vars * n_levels)
    target_tensor = torch.cat([target_tensor, forcing_input[:,1,1,:]], dim=1)
    target_tensor = target_tensor[:,None,None,:]

    return input_tensor, target_tensor


def calc_additional_loss(full_pred, reconstructed_output, input_tensor, tendency_tensor, full_output_tensor, criterion, tend_pred_rollout, tend_true_rollout, tendency_variances):
    tendency_pred_normalized = np.nan
    # Use normalized tendencies for loss (maintains gradient flow)
    # norm_tendency_loss = criterion(tendency_pred_normalized, tendency_tensor)
    norm_tendency_loss = torch.tensor(0.0, device=input_tensor.device, dtype=input_tensor.dtype)

    # calculate encoder_decoder loss
    true_decoder_loss = criterion(reconstructed_output, full_output_tensor)

    # full next state loss
    monitoring_full_state_loss = criterion(full_pred, full_output_tensor)

    # persistence loss
    # persistence_loss = criterion(input_tensor, full_output_tensor)
    persistence_loss = 0.0003

    tend = full_output_tensor - input_tensor[:, 1, 1].squeeze()

    if isinstance(criterion, nn.MSELoss):
        true_tendency_loss = torch.mean((tend_true_rollout - tend) ** 2, dim=tuple(range(tend_true_rollout.ndim - 1)))
        pred_tend_loss = torch.mean((tend_pred_rollout - tend) ** 2, dim=tuple(range(tend_pred_rollout.ndim - 1)))
    else:
        true_tendency_loss = torch.mean(torch.abs(tend_true_rollout - tend), dim=tuple(range(tend_true_rollout.ndim - 1)))
        pred_tend_loss = torch.mean(torch.abs(tend_pred_rollout - tend), dim=tuple(range(tend_pred_rollout.ndim - 1)))


    # weight tendencies by the invers of their variance
    true_tendency_loss = true_tendency_loss / tendency_variances
    pred_tend_loss = pred_tend_loss / tendency_variances
    # take the mean of each
    true_tendency_loss = true_tendency_loss.mean()
    pred_tend_loss = pred_tend_loss.mean()

    return true_decoder_loss, monitoring_full_state_loss, persistence_loss, norm_tendency_loss, true_tendency_loss, pred_tend_loss


def train_model(model, train_loader, test_loader, device, model_config_params,
                epochs=10, lr=0.001, batch_size=16384, eval_steps=108,
                patience=5, model_path="./test_model.pth", l1_target_ratio=0.075,
                warmup_eval_steps=30, warmup_start_lr=1e-6,
                scheduler_type='ReduceLROnPlateau',
                scheduler_patience=5, scheduler_factor=0.3,
                T_0=10, T_mult=1, eta_min_cosine=1e-7,
                cosine_lr_t_max_batches=10000,
                linear_decay_total_batches=0,
                linear_decay_end_lr=0.0,
                use_mixed_precision=False, loaded_checkpoint=None, resume_training=False,
                resume_optimizer_use_new_lr_config=False,
                reset_best_loss_on_load: bool = False,
                weight_decay=0.0, hydrostatic_lambda=0.0, lapse_rate_lambda=0.0, ds_pl_var_list=["Q","T","U","V","Z3"],
                print_interval_timing: bool = False, use_torch_compile: bool = False,
                grad_clip_max_norm: float = 1.0,
                noise_factor: float = 0.0,
                warmup_start_noise_factor: float = 0.0,
                final_noise_factor: float = 0.0,
                noise_std_dev_file: str = None,
                noise_surface_var_names: list = None,
                cosine_loss_weight: float = 0.0,
                use_dynamic_cosine_scaling: bool = True,
                loss_type: str = 'vicreg',
                barlow_lambda: float = 5e-3,
                loss_weights: Optional[Dict[str, float]] = None,
                mse_loss_type: str = 'mse',
                tendency_variances_path: str | Path | None = None):

    # # print_memory_usage("Start of train_model")

    if use_torch_compile:
        print("Attempting to compile the model with torch.compile()...")
        # # print_memory_usage("Before torch.compile")
        try:
            # Ensure model is on the correct device before compilation if it matters for the backend
            # model.to(device) # Usually torch.compile handles this
            compiled_model = torch.compile(model)
            # Reassign to model variable if compilation is successful
            model = compiled_model
            model.to(device) # Ensure model is on the correct device after compile
            print("Model compiled successfully.")
            # # print_memory_usage("After torch.compile success")
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}. Proceeding without compilation.")
            # # print_memory_usage("After torch.compile failure")
            # Ensure model is still on the correct device if compilation failed
            model.to(device)

    if hydrostatic_lambda > 0 or lapse_rate_lambda > 0:
        stats_path = "/glade/derecho/scratch/idavis/archive/aqua_planet/atm/hist/solin_filled/stats/"
        print("WARNING - Hydrostatic and lapse rate constraints are not fully implimented yet!")
        # print(f"Hydrostatic and lapse rate constraints are enabled with lambda values: hydrostatic={hydrostatic_lambda}, lapse_rate={lapse_rate_lambda}.")
        print(f"Using stats from {stats_path} for these constraints.")
        means = xr.open_dataset(f"{stats_path}/means.nc")
        stds = xr.open_dataset(f"{stats_path}/stds.nc")
        pressure_levels = means['lev'].values.astype(np.float32)
        # means = means.isel(lev=slice(5, None))  # Select bottom 27 levels
        # stds = stds.isel(lev=slice(5, None))  # Select bottom 27 levels
        means = means[ds_pl_var_list].to_array().values
        stds = stds[ds_pl_var_list].to_array().values
        means = means.astype(np.float32)
        stds = stds.astype(np.float32)
        residual_means = xr.open_dataset(f"{stats_path}/residual_means.nc")
        residual_stds = xr.open_dataset(f"{stats_path}/residual_stds.nc")
        residual_means = residual_means[ds_pl_var_list].to_array().values
        residual_stds = residual_stds[ds_pl_var_list].to_array().values
        residual_means = residual_means.astype(np.float32)
        residual_stds = residual_stds.astype(np.float32)

    if tendency_variances_path is None:
        raise ValueError("tendency_variances_path must be provided for offline training.")
    tendency_variances = torch.load(Path(tendency_variances_path)).to(device)
    tend_var_mean = tendency_variances.mean()
    # append the mean to the end of tendency_variances twice for the sin and cos of lat
    tendency_variances = torch.cat([tendency_variances, tend_var_mean.view(1), tend_var_mean.view(1)], dim=0)
    if mse_loss_type == 'mae':
        tendency_variances = torch.sqrt(tendency_variances)

    # --- Noise Injection Setup ---
    std_dev_profile_reshaped, std_dev_surface_reshaped = None, None
    if noise_factor > 0:
        if not noise_std_dev_file or not os.path.exists(noise_std_dev_file):
            print(f"Warning: Noise injection enabled (noise_factor={noise_factor}) but std dev file not found at '{noise_std_dev_file}'. Disabling noise.")
            noise_factor = 0.0
        elif not ds_pl_var_list or not noise_surface_var_names:
            print(f"Warning: Noise injection enabled but 'ds_pl_var_list' or 'noise_surface_var_names' is not provided. Disabling noise.")
            noise_factor = 0.0
        else:
            print(f"Noise injection enabled with factor {noise_factor}. Loading std dev from {noise_std_dev_file}")
            try:
                with xr.open_dataset(noise_std_dev_file) as noise_ds:
                    # Profile variables
                    profile_std_devs = [noise_ds[f"{var_name}_error_std"].values for var_name in ds_pl_var_list]
                    std_dev_profile = np.stack(profile_std_devs, axis=0)
                    std_dev_profile_tensor = torch.from_numpy(std_dev_profile.astype(np.float32)).to(device)
                    std_dev_profile_reshaped = std_dev_profile_tensor.view(1, -1, std_dev_profile_tensor.shape[1], 1, 1)

                    # Surface variables
                    surface_std_devs = [noise_ds[f"{var_name}_error_std"].values for var_name in noise_surface_var_names]
                    std_dev_surface = np.array(surface_std_devs)
                    std_dev_surface_tensor = torch.from_numpy(std_dev_surface.astype(np.float32)).to(device)
                    std_dev_surface_reshaped = std_dev_surface_tensor.view(1, -1, 1, 1)

                    print(f"sd_dev_profile_reshaped shape: {std_dev_profile_reshaped.shape}")
                    print(f"sd_dev_surface_reshaped shape: {std_dev_surface_reshaped.shape}")

                print("Successfully loaded and prepared noise standard deviations.")
            except Exception as e:
                print(f"Error loading or processing noise file: {e}. Disabling noise injection.")
                noise_factor = 0.0

    criterion_vicreg = VICRegLoss(  
        inv_coeff = 25.0,
        var_coeff = 18.0,
        cov_coeff = 1.0,
        gamma = 0.2,)
    
    criterion_barlow = BarlowTwinsLoss(lambda_=barlow_lambda)
    if mse_loss_type == 'mse':
        criterion = nn.MSELoss()
    elif mse_loss_type == 'mae':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown mse_loss_type: {mse_loss_type}. Must be 'mse' or 'mae'.")

    print(f"Using loss function: {loss_type.upper()}")
    if loss_type == 'barlow':
        print(f"  Barlow Twins lambda: {barlow_lambda}")
    elif loss_type == 'vicreg':
        print(f"  VICReg coefficients - inv: 25.0, var: 18.0, cov: 1.0, gamma: 0.2")
    print(f"Using base reconstruction loss: {mse_loss_type.upper()}")

    if loss_weights is None and loaded_checkpoint is not None:
        loss_weights = copy.deepcopy(loaded_checkpoint.get('hyperparams', {}).get('loss_weights'))

    configured_loss_weights = DEFAULT_LOSS_WEIGHTS.copy()
    if loss_weights:
        configured_loss_weights.update(loss_weights)
    loss_weights = configured_loss_weights

    representation_weight = loss_weights['representation']
    true_decoder_weight = loss_weights['true_decoder']
    full_state_weight = loss_weights['full_state']
    true_tendency_weight = loss_weights['true_tendency']
    pred_tendency_weight = loss_weights['pred_tendency']

    print("Using loss component weights:")
    for key in ['representation', 'true_decoder', 'full_state', 'true_tendency', 'pred_tendency']:
        print(f"  {key}: {loss_weights[key]:.4f}")

    if warmup_eval_steps > 0: init_lr = warmup_start_lr
    else: init_lr = lr
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay) # Use weight_decay
    # Corrected GradScaler initialization: removed invalid 'device' argument
    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)

    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience,
            threshold=0.001, threshold_mode='rel' # Removed verbose=True
        )
        print(f"Using ReduceLROnPlateau scheduler with patience={scheduler_patience}, factor={scheduler_factor}")
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min_cosine
        )
        print(f"Using CosineAnnealingWarmRestarts scheduler with T_0={T_0} (batches), T_mult={T_mult}, eta_min={eta_min_cosine}. Stepped per batch after warmup.")
    elif scheduler_type == 'CosineAnnealingLR':
        # T_max is now cosine_lr_t_max_batches, scheduler will be stepped per batch.
        if cosine_lr_t_max_batches <= 0:
            raise ValueError("cosine_lr_t_max_batches must be positive for CosineAnnealingLR.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_lr_t_max_batches, eta_min=eta_min_cosine
        )
        print(f"Using CosineAnnealingLR scheduler with T_max={cosine_lr_t_max_batches} (batches), eta_min={eta_min_cosine}. Stepped per batch after warmup.")
    elif scheduler_type == 'LinearDecayLR':
        if linear_decay_total_batches <= 0:
            raise ValueError("linear_decay_total_batches must be positive for LinearDecayLR.")
        if lr <= 0: # lr is the optimizer's initial LR, which is the start_lr for decay
             raise ValueError("Initial learning rate (lr) must be positive for LinearDecayLR.")
        if linear_decay_end_lr < 0:
             raise ValueError("linear_decay_end_lr cannot be negative.")
        if linear_decay_end_lr >= lr:
             print(f"Warning: linear_decay_end_lr ({linear_decay_end_lr}) >= start_lr ({lr}). This will result in LR increasing or staying constant during the decay phase.")

        # lr_lambda_linear captures 'lr', 'linear_decay_end_lr', and 'linear_decay_total_batches'
        # from the enclosing scope of train_model.
        # 'lr' from train_model acts as the starting learning rate for the decay.
        def lr_lambda_linear(current_scheduler_step):
            # 'lr' (captured) is the train_model's 'lr' argument, intended as the starting LR for the decay phase.
            # 'linear_decay_end_lr' (captured) is the target final learning rate.
            # 'linear_decay_total_batches' (captured) is the number of steps for decay.

            # Calculate the target factor for the end learning rate, relative to 'lr'.
            end_factor = linear_decay_end_lr / lr

            if current_scheduler_step >= linear_decay_total_batches:
                # If current step exceeds or meets total steps, factor should be end_factor.
                factor = end_factor
            else:
                # Linearly interpolate the factor:
                # Starts at 1.0 (meaning current LR is 'lr').
                # Ends at 'end_factor' (meaning current LR is 'linear_decay_end_lr').
                progress = float(current_scheduler_step) / float(linear_decay_total_batches)
                factor = 1.0 + (end_factor - 1.0) * progress

            return factor

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_linear)
        print(f"Using LinearDecayLR scheduler: start_lr={lr}, end_lr={linear_decay_end_lr}, total_decay_batches={linear_decay_total_batches}. Stepped per batch after warmup.")
        # Ensure scheduler.base_lrs match the 'lr' captured by the lambda.
        # 'lr' is the script's parameter for the starting LR of the decay phase.
        # Optimizer's current LR (used by LambdaLR for initial base_lrs) is init_lr.
        # If init_lr != lr (e.g. warmup active), base_lrs need adjustment.
        expected_base_lr_for_lambda = lr
        if any(b_lr != expected_base_lr_for_lambda for b_lr in scheduler.base_lrs):
            new_base_lrs = [expected_base_lr_for_lambda for _ in scheduler.base_lrs]
            scheduler.base_lrs = new_base_lrs
    else:
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")
    # --- Noise Factor Scheduling Setup ---
    # Determine the end_lr for interpolation during the decay phase
    end_lr_for_noise_schedule = -1 # Sentinel value
    if scheduler_type in ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
        end_lr_for_noise_schedule = eta_min_cosine
    elif scheduler_type == 'LinearDecayLR':
        end_lr_for_noise_schedule = linear_decay_end_lr

    gradient_norms = []
    train_losses = []
    test_losses = []
    lr_history = []
    noise_factor_history = []
    best_test_loss = float("inf")
    start_epoch = 0
    start_eval_step = 0
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    steps_no_improve = 0



    if loaded_checkpoint:
        print("Loading information from checkpoint...")
        # Load best_test_loss and best_epoch from the top level of the checkpoint
        best_test_loss = loaded_checkpoint.get('best_test_loss', float("inf"))
        best_epoch = loaded_checkpoint.get('best_epoch', -1)

        if reset_best_loss_on_load:
            print(f"Resetting best_test_loss from {best_test_loss:.6f} to infinity.")
            best_test_loss = float("inf")
            best_epoch = -1 # Also reset best_epoch

        if resume_training:
            print("Resuming training state (optimizer, scheduler, histories, steps)...")
            try:
                optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in loaded_checkpoint:
                    scheduler.load_state_dict(loaded_checkpoint['scheduler_state_dict'])
                    if scheduler_type == 'LinearDecayLR':
                        # The lr_lambda_linear for this scheduler instance was defined using the current script's 'lr'.
                        # Ensure the scheduler's base_lrs (possibly just loaded) match this 'lr'.
                        decay_start_lr_for_current_lambda = lr # 'lr' from train_model args
                        if any(b_lr != decay_start_lr_for_current_lambda for b_lr in scheduler.base_lrs):
                            new_base_lrs = [decay_start_lr_for_current_lambda for _ in scheduler.base_lrs]
                            scheduler.base_lrs = new_base_lrs
                if use_mixed_precision and 'scaler_state_dict' in loaded_checkpoint:
                    scaler.load_state_dict(loaded_checkpoint['scaler_state_dict'])
                train_losses = loaded_checkpoint.get('train_loss_history', [])
                test_losses = loaded_checkpoint.get('test_loss_history', [])
                lr_history = loaded_checkpoint.get('lr_history', [])
                noise_factor_history = loaded_checkpoint.get('noise_factor_history', [])
                # print()
                # print("WARNING: lr_history is not loaded from checkpoint, it will be empty until first eval step.")
                # print()
                start_epoch = loaded_checkpoint.get('epoch', 0)
                start_eval_step = loaded_checkpoint.get('eval_step', 0)
                steps_no_improve = loaded_checkpoint.get('steps_no_improve', 0)
                print(f"Resuming from Epoch {start_epoch}, Eval Step {start_eval_step}")
                print(f"Loaded Best Test Loss: {best_test_loss:.6f} at epoch {best_epoch}")
            except Exception as e:
                print()
                print(f"Warning: Could not load all resume states from checkpoint: {e}. Re-initializing components.")
                print()
                train_losses, test_losses, lr_history, noise_factor_history = [], [], [], []
                start_epoch, start_eval_step, steps_no_improve = 0, 0, 0
                # Re-initialize optimizer and scheduler as per script defaults if full resume fails
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                # Re-initialize scheduler based on script's scheduler_type and other params
                if scheduler_type == 'ReduceLROnPlateau':
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience)
                elif scheduler_type == 'CosineAnnealingWarmRestarts':
                    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min_cosine)
                elif scheduler_type == 'CosineAnnealingLR':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_lr_t_max_batches, eta_min=eta_min_cosine)
                elif scheduler_type == 'LinearDecayLR':
                    def lr_lambda_linear(current_scheduler_step):
                        end_factor = linear_decay_end_lr / lr
                        if current_scheduler_step >= linear_decay_total_batches:
                            factor = end_factor
                        else:
                            progress = float(current_scheduler_step) / float(linear_decay_total_batches)
                            factor = 1.0 + (end_factor - 1.0) * progress
                        return factor
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_linear)
                    # Adjust base_lrs for the re-initialized scheduler
                    expected_base_lr_for_lambda = lr
                    if any(b_lr != expected_base_lr_for_lambda for b_lr in scheduler.base_lrs):
                        new_base_lrs = [expected_base_lr_for_lambda for _ in scheduler.base_lrs]
                        scheduler.base_lrs = new_base_lrs
                else:
                    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

        elif resume_optimizer_use_new_lr_config: # New condition
            print("Resuming optimizer state and histories, but using new LR and scheduler settings from script.")
            try:
                optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully.")
                # LR and scheduler are intentionally NOT loaded from checkpoint.
                # They will be initialized/used as per current script settings (lr, scheduler_type, etc.)
                # Update optimizer's current LR to the one specified in the script (applies to all param groups)

                # Load histories, epoch, and step counters from checkpoint
                train_losses = loaded_checkpoint.get('train_loss_history', [])
                test_losses = loaded_checkpoint.get('test_loss_history', [])
                lr_history = loaded_checkpoint.get('lr_history', [])
                noise_factor_history = loaded_checkpoint.get('noise_factor_history', [])
                start_epoch = loaded_checkpoint.get('epoch', 0)
                start_eval_step = loaded_checkpoint.get('eval_step', 0)
                steps_no_improve = loaded_checkpoint.get('steps_no_improve', 0)
                print(f"Loaded histories. Resuming from Epoch {start_epoch}, Eval Step {start_eval_step}.")
                print(f"Loaded Best Test Loss from checkpoint: {best_test_loss:.6f} at epoch {best_epoch}")

                if warmup_eval_steps > 0:
                    init_lr = warmup_start_lr
                    warmup_eval_steps += start_eval_step # Adjust warmup steps to account for resume
                else: init_lr = lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = init_lr
                print(f"Initial Optimizer LR set to script value: {init_lr}")

            except Exception as e:
                print(f"Warning: Could not load optimizer state or histories from checkpoint: {e}. Re-initializing optimizer and histories.")
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                train_losses, test_losses, lr_history, noise_factor_history = [], [], [], []
                start_epoch, start_eval_step, steps_no_improve = 0, 0, 0

        else: # This is the case for load=True, resume_training=False, resume_optimizer_use_new_lr_config=False
            print("Starting fresh training session (model weights loaded, but optimizer/scheduler/histories are new). Best loss from ckpt retained.")
            # Optimizer and scheduler are already initialized with script defaults earlier.
            # Ensure LR is set to the script's current LR value for the fresh optimizer.
            if warmup_eval_steps > 0: init_lr = warmup_start_lr
            else: init_lr = lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = init_lr
            train_losses, test_losses, lr_history, noise_factor_history = [], [], [], []
            start_epoch, start_eval_step, steps_no_improve = 0, 0, 0
            # best_model_state is already a deepcopy of the loaded model.state_dict() if load was successful
            # or a fresh model if load failed/was false.

    print("Training HybridCNNTransformerSCAM model...")
    if use_mixed_precision:
        print("Using mixed precision training")

    running_train_loss = 0.0
    running_train_variance = 0.0
    running_train_invariance = 0.0
    running_train_covariance = 0.0
    running_train_output_std = 0.0
    running_train_output_magnitude = 0.0
    running_train_tendency_loss = 0.0
    running_train_true_tendency_loss = 0.0
    running_train_pred_tendency_loss = 0.0
    running_train_true_decoder_loss = 0.0
    running_train_monitoring_full_state_loss = 0.0
    running_train_monitoring_tendency_loss = 0.0
    running_train_persistence_loss = 0.0
    running_train_samples = 0
    eval_step_counter = start_eval_step
    n_evals_done = start_eval_step // eval_steps if eval_steps > 0 else 0

    # Initialize timer for the first interval's training part
    interval_start_time = time.time()
    train_data_loading_time_in_interval = 0.0
    last_batch_end_time = time.time() # For data loading timing

    early_stop_triggered = False
    for epoch in tqdm.tqdm(range(start_epoch, epochs), desc="Training Epochs"):
        if early_stop_triggered:
            break
        model.train()
        # # print_memory_usage(f"Epoch {epoch} Start")

        for i, data_chunk in enumerate(train_loader):
            # --- Data Loading Time Measurement ---
            batch_load_start_time = time.time()
            train_data_loading_time_in_interval += batch_load_start_time - last_batch_end_time

            if early_stop_triggered:
                break
            # # print_memory_usage(f"Epoch {epoch}, Chunk {i} Loaded")

            # inputs_tuple, targets_tuple = data_chunk
            # # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, After unpacking data_chunk from DataLoader")

            # # Squeeze the DataLoader's batch dimension (assuming DataLoader batch_size is 1)
            # # and move to device immediately.
            # profile_input_chunk_content = inputs_tuple[0].squeeze(0).to(device)#[:,:,6:,:,:]
            # # surface_input_chunk_content = inputs_tuple[1].squeeze(0).to(device)
            # forcing_input_chunk_content = inputs_tuple[2].squeeze(0).to(device)

            # profile_target_chunk_content = targets_tuple[0].squeeze(0).to(device)#[:,:,6:]
            # surface_target_chunk_content = targets_tuple[1].squeeze(0).to(device)
            # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, After creating *_chunk_content and .to(device)")

            profile_input = data_chunk['profile_input'].squeeze(0).to(device)
            forcing_input = data_chunk['forcing_input'].squeeze(0).to(device).to(dtype=torch.float32)
            profile_target = data_chunk['profile_target'].squeeze(0).to(device)


            input_tensor, target_tensor = reshape_for_mixer(profile_input, forcing_input, profile_target)

            # # AGGRESSIVE DELETION of original CPU tensors from DataLoader
            # del inputs_tuple
            # del targets_tuple
            # # data_chunk itself is more complex as it's the iterator variable,
            # but its components (inputs_tuple, targets_tuple) are the big ones.
            # We will still del data_chunk at the end of this outer loop iteration.
            # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, After explicit del of inputs_tuple & targets_tuple")
            gc.collect() # Force garbage collection immediately after deleting CPU tuples
            # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, After gc.collect (post CPU tuple del)")

            num_samples_in_chunk = input_tensor.size(0) # Number of samples in the current file chunk
            # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, Before mini-batch loop")

            # Shuffle the chunk content before creating mini-batches
            if num_samples_in_chunk > 0:
                permuted_indices = torch.randperm(num_samples_in_chunk, device=device)
                input_tensor = input_tensor[permuted_indices]
                target_tensor = target_tensor[permuted_indices]

            for start_idx in range(0, num_samples_in_chunk, batch_size):
                if early_stop_triggered: break
                end_idx = min(start_idx + batch_size, num_samples_in_chunk)
                # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, Mini-batch {start_idx//batch_size}, Before zero_grad")

                optimizer.zero_grad(set_to_none=True)
                # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, Mini-batch {start_idx//batch_size}, After zero_grad")

                # Corrected autocast call: use 'device_type'
                with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):
                    # Create mini-batches from the squeezed chunk content
                    mini_batch_in = input_tensor[start_idx:end_idx]
                    mini_batch_target = target_tensor[start_idx:end_idx].squeeze()


                    # --- Noise Factor Scheduling ---
                    current_lr_for_noise_calc = optimizer.param_groups[0]['lr']
                    current_noise_factor = 0.0
                    is_in_warmup_phase_for_noise = warmup_eval_steps > 0 and eval_step_counter <= warmup_eval_steps

                    if noise_factor > 0:
                        if is_in_warmup_phase_for_noise:
                            # Interpolate noise factor during warmup phase
                            # Progress is based on LR, from warmup_start_lr to lr
                            if lr > warmup_start_lr:
                                progress = (current_lr_for_noise_calc - warmup_start_lr) / (lr - warmup_start_lr)
                                current_noise_factor = warmup_start_noise_factor + (noise_factor - warmup_start_noise_factor) * progress
                            else:
                                current_noise_factor = noise_factor # Avoid division by zero if lrs are same
                        else:
                            # Interpolate noise factor during decay phase
                            # Progress is based on LR, from lr to end_lr_for_noise_schedule
                            if lr > end_lr_for_noise_schedule:
                                progress = (lr - current_lr_for_noise_calc) / (lr - end_lr_for_noise_schedule)
                                current_noise_factor = noise_factor + (final_noise_factor - noise_factor) * progress
                            else:
                                current_noise_factor = final_noise_factor # Avoid division by zero

                        # Clamp to ensure the value stays within the defined bounds
                        min_noise = min(warmup_start_noise_factor, noise_factor, final_noise_factor)
                        max_noise = max(warmup_start_noise_factor, noise_factor, final_noise_factor)
                        current_noise_factor = max(min_noise, min(current_noise_factor, max_noise))

                    # --- Apply Noise Injection (if enabled) ---
                    if current_noise_factor > 0 and std_dev_profile_reshaped is not None and std_dev_surface_reshaped is not None:
                        # print shapes
                        # print(f"profile_in shape: {mini_batch_profile_in.shape}, std_dev_profile_reshaped shape: {std_dev_profile_reshaped.shape}")
                        # print(f"surface_in shape: {mini_batch_surface_in.shape}, std_dev_surface_reshaped shape: {std_dev_surface_reshaped.shape}")
                        profile_noise = torch.randn_like(mini_batch_in)
                        # surface_noise = torch.randn_like(mini_batch_surface_in)
                        mini_batch_in += profile_noise * (current_noise_factor * std_dev_profile_reshaped)
                        # mini_batch_surface_in += surface_noise * (current_noise_factor * std_dev_surface_reshaped)

                    (
                        mixer_pred,
                        encoder_target,
                        full_pred,
                        reconstructed_output,
                        tend_pred_rollout,
                        tend_true_rollout,
                    ) = model(
                        mini_batch_in,
                        mini_batch_target,
                        train=True,
                        global_mode=False,
                    )

                    full_output_tensor = mini_batch_target
                    tendency_tensor = full_output_tensor - mini_batch_in[:,1,1]

                    if tend_pred_rollout is None:
                        tend_pred_rollout = torch.zeros_like(full_output_tensor)
                    if tend_true_rollout is None:
                        tend_true_rollout = torch.zeros_like(full_output_tensor)

                    # Calculate loss based on selected type
                    if loss_type == 'vicreg':
                        loss_dict = criterion_vicreg(
                            mixer_pred.view(-1, mixer_pred.shape[-1]),
                            encoder_target.view(-1, encoder_target.shape[-1])
                        )
                        loss = loss_dict["loss"]
                        variance_loss = loss_dict["var-loss"].item()
                        invariance_loss = loss_dict["inv-loss"].item()
                        covariance_loss = loss_dict["cov-loss"].item()
                    elif loss_type == 'barlow':
                        loss = criterion_barlow(
                            mixer_pred.view(-1, mixer_pred.shape[-1]),
                            encoder_target.view(-1, encoder_target.shape[-1])
                        )
                        # For Barlow Twins, calculate variance and invariance manually for diagnostics
                        with torch.no_grad():
                            mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1])
                            if mixer_pred_flat.size(0) > 1:
                                variance_loss = torch.var(mixer_pred_flat, dim=0, unbiased=False).mean().item()
                            else:
                                variance_loss = 0.0
                        invariance_loss = criterion(mixer_pred, encoder_target).item()
                        covariance_loss = 0.0  # Not directly comparable to VICReg covariance
                    else:
                        # MSE_decoded = criterion(full_pred, full_output_tensor)
                        MSE_encoded = criterion(mixer_pred, encoder_target)
                        loss = MSE_encoded
                        variance_loss = 0.0
                        invariance_loss = MSE_encoded.item()
                        covariance_loss = 0.0

                    (
                    true_decoder_loss,
                    monitoring_full_state_loss,
                    persistence_loss,
                    norm_tendency_loss,
                    true_tendency_loss,
                    pred_tend_loss,
                    ) = calc_additional_loss(
                    full_pred,
                    reconstructed_output,
                    mini_batch_in,
                    tendency_tensor,
                    full_output_tensor,
                    criterion,
                    tend_pred_rollout,
                    tend_true_rollout,
                    tendency_variances,
                    )

                    true_decoder_loss_val = true_decoder_loss.detach()
                    monitoring_full_state_loss_val = monitoring_full_state_loss.detach()
                    norm_tendency_loss_val = norm_tendency_loss.detach()
                    persistence_loss_val = float(persistence_loss)
                    true_tendency_loss_val = true_tendency_loss.detach()
                    pred_tend_loss_val = pred_tend_loss.detach()

                    # normalize losses by their magnitudes
                    representation_loss_val = loss.detach()
                    true_decoder_loss_norm = true_decoder_loss / (true_decoder_loss_val.abs() + 1e-10)
                    full_state_norm = monitoring_full_state_loss / (monitoring_full_state_loss_val.abs() + 1e-10)
                    true_tend_norm = true_tendency_loss / (true_tendency_loss_val.abs() + 1e-10)
                    pred_tend_norm = pred_tend_loss / (pred_tend_loss_val.abs() + 1e-10)
                    loss_norm = loss / (representation_loss_val + 1e-10)

                    loss_backprop = (
                        representation_weight * loss_norm
                        + true_decoder_weight * true_decoder_loss_norm
                        + full_state_weight * full_state_norm
                        + true_tendency_weight * true_tend_norm
                        + pred_tendency_weight * pred_tend_norm
                    )

                    weighted_loss_value = (
                        representation_weight * representation_loss_val
                        + true_decoder_weight * true_decoder_loss_val
                        + full_state_weight * monitoring_full_state_loss_val
                        + true_tendency_weight * true_tendency_loss_val
                        + pred_tendency_weight * pred_tend_loss_val
                    )

                    # Calculate output statistics (same for both loss types)
                    mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1])
                    if mixer_pred_flat.size(0) > 1:
                        output_std = torch.std(mixer_pred_flat, dim=0, unbiased=False).mean().item()
                    else:
                        output_std = 0.0
                    output_magnitude = torch.norm(mixer_pred_flat, dim=1).mean().item()

                    current_mini_batch_size = mini_batch_in.size(0)

                    if l1_target_ratio > 0:
                        loss_val = weighted_loss_value.item()
                        l1_raw_penalty = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                        if l1_raw_penalty.item() > 1e-10:
                            adaptive_l1_lambda = l1_target_ratio * loss_val / l1_raw_penalty.item()
                            l1_loss = adaptive_l1_lambda * l1_raw_penalty
                            loss = loss + l1_loss

                # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, Mini-batch {start_idx//batch_size}, Before loss.backward")
                if use_mixed_precision:
                    scaler.scale(loss_backprop).backward()
                    scaler.unscale_(optimizer)  # Unscale before clipping

                    # Calculate and record the gradient norm before clipping
                    if grad_clip_max_norm > 0:
                        # # Calculate total norm
                        # total_norm = torch.norm(
                        #     torch.stack([torch.norm(p.grad.detach(), 2)
                        #                 for p in model.parameters()
                        #                 if p.grad is not None]),
                        #     2).item()
                        # gradient_norms.append(total_norm)

                        # Apply clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_backprop.backward()

                    # Calculate and record the gradient norm before clipping
                    if grad_clip_max_norm > 0:
                        # Calculate total norm
                        # total_norm = torch.norm(
                        #     torch.stack([torch.norm(p.grad.detach(), 2)
                        #                 for p in model.parameters()
                        #                 if p.grad is not None]),
                        #     2).item()
                        # gradient_norms.append(total_norm)

                        # Apply clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

                    optimizer.step()

                running_train_loss += weighted_loss_value.item() 
                running_train_variance += variance_loss 
                running_train_invariance += invariance_loss
                running_train_covariance += covariance_loss 
                running_train_output_std += output_std 
                running_train_output_magnitude += output_magnitude 
                running_train_tendency_loss += norm_tendency_loss_val
                running_train_true_decoder_loss += true_decoder_loss_val
                running_train_pred_tendency_loss += pred_tend_loss_val
                running_train_true_tendency_loss += true_tendency_loss_val
                running_train_monitoring_full_state_loss += monitoring_full_state_loss_val
                running_train_persistence_loss += persistence_loss_val
                running_train_samples += 1
                eval_step_counter += 1

                del mixer_pred, encoder_target, full_pred, reconstructed_output, mixer_pred_flat
                del true_decoder_loss, monitoring_full_state_loss, norm_tendency_loss, loss
                del full_output_tensor, tendency_tensor, tend_pred_rollout, tend_true_rollout
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                if eval_step_counter == 3:
                    print()
                    print("Train Loss of first 4 batches:", running_train_loss / running_train_samples)
                    # Add component breakdown
                    if loss_type == 'vicreg':
                        print(f"  Components - Var: {running_train_variance / running_train_samples:.4f}, Inv: {running_train_invariance / running_train_samples:.4f}, Cov: {running_train_covariance / running_train_samples:.4f}")
                    elif loss_type == 'barlow':
                        print(f"  Diagnostics - Var: {running_train_variance / running_train_samples:.4f}, Inv ({mse_loss_type.upper()}): {running_train_invariance / running_train_samples:.4f}")
                    print(f"  Output Stats - Std: {running_train_output_std / running_train_samples:.4f}, Magnitude: {running_train_output_magnitude / running_train_samples:.4f}")
                    print(f"  Norm Tendency Loss: {running_train_tendency_loss / running_train_samples:.4f}, True Tendency Loss: {running_train_true_tendency_loss / running_train_samples:.4f}, Pred Tendency Loss: {running_train_pred_tendency_loss / running_train_samples:.4f}, Encoder-Decoder Loss: {running_train_true_decoder_loss / running_train_samples:.4f}")
                    print(f"  Diagnostic Losses - Full State: {running_train_monitoring_full_state_loss / running_train_samples:.4f}, Persistence: {running_train_persistence_loss / running_train_samples:.4f}")

                is_in_warmup_phase = warmup_eval_steps > 0 and eval_step_counter <= warmup_eval_steps

                if is_in_warmup_phase:
                    # Cosine warmup
                    # if resume_optimizer_use_new_lr_config then appropriately scale variables so warmup happens correctly
                    if resume_optimizer_use_new_lr_config: progress = (eval_step_counter -start_eval_step)  / (warmup_eval_steps-start_eval_step)
                    else: progress = eval_step_counter / warmup_eval_steps
                    # Cosine curve from 0 to 1: 0.5 * (1 - cos(progress * pi))
                    # This makes LR stay lower for longer then ramp up faster.
                    cosine_factor = 0.5 * (1 - math.cos(progress * math.pi))
                    current_lr = warmup_start_lr + (lr - warmup_start_lr) * cosine_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                else: # After warmup phase
                    # Step schedulers that are updated per batch
                    if isinstance(scheduler, (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                              torch.optim.lr_scheduler.CosineAnnealingLR,
                                              torch.optim.lr_scheduler.LambdaLR)): # Added LambdaLR
                        scheduler.step()

                # --- End of Batch ---
                # Update timer for data loading measurement
                last_batch_end_time = time.time()

                if eval_step_counter % eval_steps == 0 and running_train_samples > 0:
                    # End of training for the current interval, start of evaluation specific timing
                    current_time_before_eval = time.time()
                    # print_memory_usage(f"Eval Cycle {n_evals_done+1}, Before evaluate_loss")

                    n_evals_done +=1
                    # Calculate training loss for reporting
                    avg_train_loss = running_train_loss / running_train_samples
                    avg_train_variance = running_train_variance / running_train_samples
                    avg_train_invariance = running_train_invariance / running_train_samples
                    avg_train_covariance = running_train_covariance / running_train_samples
                    avg_train_output_std = running_train_output_std / running_train_samples
                    avg_train_output_magnitude = running_train_output_magnitude / running_train_samples
                    avg_train_tendency_loss = running_train_tendency_loss / running_train_samples
                    avg_train_true_tendency_loss = running_train_true_tendency_loss / running_train_samples
                    avg_train_true_decoder_loss = running_train_true_decoder_loss / running_train_samples
                    avg_train_pred_tendency_loss = running_train_pred_tendency_loss / running_train_samples
                    avg_train_monitoring_full_state_loss = running_train_monitoring_full_state_loss / running_train_samples
                    avg_train_monitoring_tendency_loss = running_train_monitoring_tendency_loss / running_train_samples
                    avg_train_persistence_loss = running_train_persistence_loss / running_train_samples
                    train_losses.append(avg_train_loss)

                    current_eval_lr = optimizer.param_groups[0]['lr']
                    lr_history.append(current_eval_lr)
                    noise_factor_history.append(current_noise_factor)

                    print(f"--- Eval Cycle {n_evals_done} (Epoch {epoch}, Step {eval_step_counter}) ---")
                    print(f"  Train Loss: {avg_train_loss:.6f}")
                    if loss_type == 'vicreg':
                        print(f"    Components - Var: {avg_train_variance:.4f}, Inv: {avg_train_invariance:.4f}, Cov: {avg_train_covariance:.4f}")
                    elif loss_type == 'barlow':
                        print(f"    Diagnostics - Var: {avg_train_variance:.4f}, Inv ({mse_loss_type.upper()}): {avg_train_invariance:.4f}")
                    print(f"    Output Stats - Std: {avg_train_output_std:.4f}, Magnitude: {avg_train_output_magnitude:.4f}")
                    print(
                        f"    calc_additional_loss - Norm Tendency: {avg_train_tendency_loss:.4f}, "
                        f"True Tendency: {avg_train_true_tendency_loss:.4f}, Pred Tendency: {avg_train_pred_tendency_loss:.4f}, Decoder: {avg_train_true_decoder_loss:.4f}"
                    )
                    print(
                        f"    Monitoring Losses - Tendency: {avg_train_monitoring_tendency_loss:.4f}, "
                        f"Full State: {avg_train_monitoring_full_state_loss:.4f}, Persistence: {avg_train_persistence_loss:.4f}"
                    )

                    eval_start_time = time.time()
                    avg_test_loss, test_data_loading_time = evaluate_loss(
                        model,
                        test_loader,
                        device,
                        2048*16,
                        use_mixed_precision,
                        loss_type=loss_type,
                        barlow_lambda=barlow_lambda,
                        tendency_variances=tendency_variances,
                        loss_weights=loss_weights,
                        mse_loss_type=mse_loss_type,
                    )
                    evaluation_duration_for_interval = time.time() - eval_start_time
                    test_losses.append(avg_test_loss)
                    
                    print(f"  Test Loss:  {avg_test_loss:.6f}")
                    print(f"  LR:         {current_eval_lr:.2e}")
                    if noise_factor > 0:
                        print(f"  Noise Factor: {current_noise_factor:.2e}")
                    
                    if print_interval_timing:
                        training_duration_for_interval = current_time_before_eval - interval_start_time
                        print(f"  Timing: Training part = {training_duration_for_interval:.2f}s (Data Loading: {train_data_loading_time_in_interval:.2f}s)")
                        print(f"          Evaluation part = {evaluation_duration_for_interval:.2f}s (Data Loading: {test_data_loading_time:.2f}s)")
                    print(f"--- End Eval Cycle {n_evals_done} ---")

                    # Reset timer for the next interval's operations (starts with training)
                    interval_start_time = time.time()
                    train_data_loading_time_in_interval = 0.0

                    running_train_loss = 0.0
                    running_train_variance = 0.0
                    running_train_invariance = 0.0
                    running_train_covariance = 0.0
                    running_train_output_std = 0.0
                    running_train_output_magnitude = 0.0
                    running_train_tendency_loss = 0.0
                    running_train_true_tendency_loss = 0.0
                    running_train_pred_tendency_loss = 0.0
                    running_train_true_decoder_loss = 0.0
                    running_train_monitoring_full_state_loss = 0.0
                    running_train_monitoring_tendency_loss = 0.0
                    running_train_persistence_loss = 0.0
                    running_train_samples = 0

                    # if gradient_norms:
                    #     norms_tensor = torch.tensor(gradient_norms)
                    #     mean_norm = norms_tensor.mean().item()
                    #     std_norm = norms_tensor.std().item()
                    #     min_norm = norms_tensor.min().item()
                    #     max_norm = norms_tensor.max().item()
                    #     median_norm = torch.median(norms_tensor).item()
                    #     clipped_pct = (norms_tensor > grad_clip_max_norm).float().mean().item() * 100 if grad_clip_max_norm > 0 else 0

                    #     print(f"  Gradient Norm Stats (over {len(gradient_norms)} batches):")
                    #     print(f"    Mean: {mean_norm:.4f}, Std: {std_norm:.4f}")
                    #     print(f"    Min: {min_norm:.4f}, Max: {max_norm:.4f}, Median: {median_norm:.4f}")
                    #     print(f"    Clipped: {clipped_pct:.1f}% of batches")

                    #     # Reset tracking for next period
                    #     gradient_norms = []

                    if not is_in_warmup_phase and isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(avg_test_loss)

                    if avg_test_loss < best_test_loss:
                        best_test_loss = avg_test_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        steps_no_improve = 0
                        if model_path:
                            # # print_memory_usage(f"Eval Cycle {n_evals_done}, Before saving model")
                            torch.save({
                                'epoch': epoch,
                                'eval_step': eval_step_counter,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'scaler_state_dict': scaler.state_dict() if use_mixed_precision else None,
                                'best_test_loss': best_test_loss,
                                'best_epoch': best_epoch,
                                'train_loss_history': train_losses,
                                'test_loss_history': test_losses,
                                'lr_history': lr_history,
                                'noise_factor_history': noise_factor_history,
                                'steps_no_improve': steps_no_improve,
                                'hyperparams': {
                                    'model_config_params': model_config_params,
                                    'architecture': type(model).__name__,
                                    'lr': lr,
                                    'batch_size': batch_size,
                                    'l1_target_ratio': l1_target_ratio,
                                    'epochs_trained_at_save': epoch,
                                    'warmup_eval_steps': warmup_eval_steps,
                                    'warmup_start_lr': warmup_start_lr,
                                    'scheduler_type': scheduler_type,
                                    'scheduler_patience': scheduler_patience,
                                    'scheduler_factor': scheduler_factor,
                                    'T_0': T_0,
                                    'T_mult': T_mult,
                                    'eta_min_cosine': eta_min_cosine,
                                    'cosine_lr_t_max_batches': cosine_lr_t_max_batches,
                                    'linear_decay_total_batches': linear_decay_total_batches,
                                    'linear_decay_end_lr': linear_decay_end_lr,
                                    'weight_decay': weight_decay,
                                    'grad_clip_max_norm': grad_clip_max_norm,
                                    'loss_type': loss_type,
                                    'barlow_lambda': barlow_lambda,
                                    'loss_weights': copy.deepcopy(loss_weights),
                                    'mse_loss_type': mse_loss_type,
                                }
                            }, model_path)
                            print(f"New best model saved to {model_path} (Test Loss: {best_test_loss:.6f} at Epoch {best_epoch}, Step {eval_step_counter})")
                            # # print_memory_usage(f"Eval Cycle {n_evals_done}, After saving model")
                    else:
                        steps_no_improve += 1
                        print(f"No improvement in test loss for {steps_no_improve} evaluation cycles.")

                    if steps_no_improve >= patience:
                        print(f"Early stopping triggered after {patience} evaluation cycles without improvement.")
                        early_stop_triggered = True
                        break
            if early_stop_triggered:
                break

            # Explicitly delete references to large chunk tensors

            # inputs_tuple and targets_tuple were deleted earlier
            del data_chunk # Also delete the original data_chunk reference from DataLoader
            # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, After deleting chunk tensors and data_chunk")

            # Force garbage collection
            gc.collect()
            # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, After gc.collect()")
            if device.type == 'cuda': # If you also want to be aggressive with GPU cache (optional here)
                torch.cuda.empty_cache()
                # # print_memory_usage(f"Epoch {epoch}, Chunk {i}, After cuda.empty_cache()")

        if early_stop_triggered:
            break
        # # print_memory_usage(f"Epoch {epoch} End")

    print("Training finished.")
    # # print_memory_usage("End of train_model")
    if best_epoch != -1:
        print(f"Best model had Test Loss: {best_test_loss:.6f} (achieved at Epoch {best_epoch})")
        model.load_state_dict(best_model_state)
    else:
        print("No improvement over initial state or training was too short. Returning current model state.")

    return train_losses, test_losses, lr_history, noise_factor_history, model

def evaluate_loss(model, data_loader, device, eval_batch_size, use_mixed_precision=False, max_files_to_evaluate: int = 32, loss_type: str = 'vicreg', barlow_lambda: float = 5e-3, tendency_variances=None, loss_weights: Optional[Dict[str, float]] = None, mse_loss_type: str = 'mse'):
    if tendency_variances is None:
        raise ValueError("tendency_variances must be provided to evaluate_loss to match training loss computation.")

    configured_loss_weights = DEFAULT_LOSS_WEIGHTS.copy()
    if loss_weights:
        configured_loss_weights.update(loss_weights)
    loss_weights = configured_loss_weights

    representation_weight = loss_weights['representation']
    true_decoder_weight = loss_weights['true_decoder']
    full_state_weight = loss_weights['full_state']
    true_tendency_weight = loss_weights['true_tendency']
    pred_tendency_weight = loss_weights['pred_tendency']
    model.eval()
    accumulated_loss = 0.0
    accumulated_samples = 0
    
    # Track components separately
    accumulated_variance = 0.0
    accumulated_invariance = 0.0
    accumulated_covariance = 0.0
    accumulated_output_std = 0.0
    accumulated_output_magnitude = 0.0

    accumulated_norm_tendency_loss = 0.0
    accumulated_true_tendency_loss = 0.0
    accumulated_pred_tendency_loss = 0.0
    accumulated_true_decoder_loss = 0.0
    accumulated_monitoring_full_state_loss = 0.0
    accumulated_persistence_loss = 0.0
    
    files_processed_count = 0
    criterion_vicreg_eval = VICRegLoss()
    criterion_barlow_eval = BarlowTwinsLoss(lambda_=barlow_lambda)
    if mse_loss_type == 'mse':
        criterion = nn.MSELoss()
    elif mse_loss_type == 'mae':
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown mse_loss_type: {mse_loss_type}. Must be 'mse' or 'mae'.")
    total_data_loading_time = 0.0
    last_batch_end_time = time.time()
    
    with torch.no_grad():
        for i, data_chunk in enumerate(data_loader):
            batch_load_start_time = time.time()
            total_data_loading_time += batch_load_start_time - last_batch_end_time
            if max_files_to_evaluate > 0 and files_processed_count >= max_files_to_evaluate:
                break

            profile_input = data_chunk['profile_input'].squeeze(0)
            forcing_input = data_chunk['forcing_input'].squeeze(0).to(dtype=torch.float32)
            profile_target = data_chunk['profile_target'].squeeze(0)

            input_chunk, target_chunk = reshape_for_mixer(profile_input, forcing_input, profile_target)

            tendency_chunk = target_chunk - input_chunk

            num_samples_in_chunk = input_chunk.size(0)
            if num_samples_in_chunk == 0:
                continue

            for start_idx in range(0, num_samples_in_chunk, eval_batch_size):
                end_idx = min(start_idx + eval_batch_size, num_samples_in_chunk)

                mini_batch_input = input_chunk[start_idx:end_idx].to(device, non_blocking=True)
                mini_batch_target = target_chunk[start_idx:end_idx].to(device, non_blocking=True).squeeze()
                mini_batch_tendency = tendency_chunk[start_idx:end_idx].to(device, non_blocking=True)

                current_eval_batch_size = mini_batch_input.size(0)
                if current_eval_batch_size == 0:
                    continue

                with torch.amp.autocast(device_type=device.type, enabled=use_mixed_precision):

                    (
                        mixer_pred,
                        encoder_target,
                        full_pred,
                        reconstructed_output,
                        tend_pred_rollout,
                        tend_true_rollout,
                    ) = model(
                        mini_batch_input,
                        mini_batch_target,
                        train=True,
                        global_mode=False,
                    )

                    if tend_pred_rollout is None:
                        tend_pred_rollout = torch.zeros_like(mini_batch_target)
                    if tend_true_rollout is None:
                        tend_true_rollout = torch.zeros_like(mini_batch_target)
                    

                    (
                        true_decoder_loss,
                        monitoring_full_state_loss,
                        persistence_loss,
                        norm_tendency_loss,
                        true_tendency_loss,
                        pred_tend_loss,
                    ) = calc_additional_loss(
                        full_pred,
                        reconstructed_output,
                        mini_batch_input,
                        mini_batch_tendency,
                        mini_batch_target,
                        criterion,
                        tend_pred_rollout,
                        tend_true_rollout,
                        tendency_variances,
                    )

                    # Calculate loss based on selected type
                    if loss_type == 'vicreg':
                        loss_dict = criterion_vicreg_eval(
                            mixer_pred.view(-1, mixer_pred.shape[-1]),
                            encoder_target.view(-1, encoder_target.shape[-1])
                        )
                        loss = loss_dict["loss"]
                        variance_loss = loss_dict["var-loss"].item()
                        invariance_loss = loss_dict["inv-loss"].item()
                        covariance_loss = loss_dict["cov-loss"].item()
                    elif loss_type == 'barlow':
                        loss = criterion_barlow_eval(
                            mixer_pred.view(-1, mixer_pred.shape[-1]),
                            encoder_target.view(-1, encoder_target.shape[-1])
                        )
                        mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1])
                        if mixer_pred_flat.size(0) > 1:
                            variance_loss = torch.var(mixer_pred_flat, dim=0, unbiased=False).mean().item()
                        else:
                            variance_loss = 0.0
                        invariance_loss = criterion(mixer_pred, encoder_target).item()
                        covariance_loss = 0.0
                    else:
                        # MSE_decoded = criterion(full_pred, mini_batch_target)
                        MSE_encoded = criterion(mixer_pred, encoder_target)
                        loss = MSE_encoded
                        variance_loss = 0.0
                        invariance_loss = MSE_encoded.item()
                        covariance_loss = 0.0

                    true_decoder_loss_val = true_decoder_loss.detach()
                    monitoring_full_state_loss_val = monitoring_full_state_loss.detach()
                    norm_tendency_loss_val = norm_tendency_loss.detach()
                    true_tendency_loss_val = true_tendency_loss.detach()
                    pred_tend_loss_val = pred_tend_loss.detach()

                    # true_decoder_loss_norm = true_decoder_loss / (true_decoder_loss_detached.abs() + 1e-10)
                    # full_state_norm = monitoring_full_state_loss / (monitoring_full_state_loss_detached.abs() + 1e-10)
                    # true_tend_norm = true_tendency_loss / (true_tendency_loss_detached.abs() + 1e-10)
                    # pred_tend_norm = pred_tend_loss / (pred_tend_loss_detached.abs() + 1e-10)
                    # loss = loss / (loss.detach() + 1e-10)
                    representation_loss_val = loss.detach()
                    weighted_loss_val = (
                        representation_weight * representation_loss_val
                        + true_decoder_weight * true_decoder_loss_val
                        + full_state_weight * monitoring_full_state_loss_val
                        + true_tendency_weight * true_tendency_loss_val
                        + pred_tendency_weight * pred_tend_loss_val
                    )
                    persistence_loss_val = float(persistence_loss)


                    mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1])
                    if mixer_pred_flat.size(0) > 1:
                        output_std = torch.std(mixer_pred_flat, dim=0, unbiased=False).mean().item()
                    else:
                        output_std = 0.0
                    output_magnitude = torch.norm(mixer_pred_flat, dim=1).mean().item()

                    accumulated_variance += variance_loss * current_eval_batch_size
                    accumulated_invariance += invariance_loss * current_eval_batch_size
                    accumulated_covariance += covariance_loss * current_eval_batch_size
                    accumulated_output_std += output_std * current_eval_batch_size
                    accumulated_output_magnitude += output_magnitude * current_eval_batch_size

                    accumulated_norm_tendency_loss += norm_tendency_loss_val * current_eval_batch_size
                    accumulated_true_tendency_loss += true_tendency_loss_val * current_eval_batch_size
                    accumulated_pred_tendency_loss += pred_tend_loss_val * current_eval_batch_size
                    accumulated_true_decoder_loss += true_decoder_loss_val * current_eval_batch_size
                    accumulated_monitoring_full_state_loss += monitoring_full_state_loss_val * current_eval_batch_size
                    accumulated_persistence_loss += persistence_loss_val * current_eval_batch_size
                    accumulated_loss += weighted_loss_val.item() * current_eval_batch_size
                    accumulated_samples += current_eval_batch_size

                del mini_batch_input, mini_batch_target, mini_batch_tendency
                del mixer_pred, encoder_target, full_pred, reconstructed_output, mixer_pred_flat

            del input_chunk, target_chunk, tendency_chunk, profile_input, forcing_input, profile_target, data_chunk
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            files_processed_count += 1
            last_batch_end_time = time.time()

    if accumulated_samples > 0:
        avg_loss = accumulated_loss / accumulated_samples
        avg_variance = accumulated_variance / accumulated_samples
        avg_invariance = accumulated_invariance / accumulated_samples
        avg_covariance = accumulated_covariance / accumulated_samples
        avg_output_std = accumulated_output_std / accumulated_samples
        avg_output_magnitude = accumulated_output_magnitude / accumulated_samples
        avg_norm_tendency_loss = accumulated_norm_tendency_loss / accumulated_samples
        avg_true_tendency_loss = accumulated_true_tendency_loss / accumulated_samples
        avg_pred_tendency_loss = accumulated_pred_tendency_loss / accumulated_samples
        avg_true_decoder_loss = accumulated_true_decoder_loss / accumulated_samples
        avg_monitoring_full_state_loss = accumulated_monitoring_full_state_loss / accumulated_samples
        avg_persistence_loss = accumulated_persistence_loss / accumulated_samples
    else:
        avg_loss = 0.0
        avg_variance = 0.0
        avg_invariance = 0.0
        avg_covariance = 0.0
        avg_output_std = 0.0
        avg_output_magnitude = 0.0
        avg_norm_tendency_loss = 0.0
        avg_true_tendency_loss = 0.0
        avg_pred_tendency_loss = 0.0
        avg_true_decoder_loss = 0.0
        avg_monitoring_full_state_loss = 0.0
        avg_persistence_loss = 0.0

    if loss_type == 'vicreg':
        print(f"    Test Loss Components - Var: {avg_variance:.4f}, Inv: {avg_invariance:.4f}, Cov: {avg_covariance:.4f}")
    elif loss_type == 'barlow':
        print(f"    Test Loss Diagnostics - Var: {avg_variance:.4f}, Inv ({mse_loss_type.upper()}): {avg_invariance:.4f}")
    print(f"    Test Output Stats - Std: {avg_output_std:.4f}, Magnitude: {avg_output_magnitude:.4f}")
    print(
        f"    Test calc_additional_loss - Norm Tendency: {avg_norm_tendency_loss:.4f}, "
        f"True Tendency: {avg_true_tendency_loss:.4f}, Pred Tendency: {avg_pred_tendency_loss:.4f}, Decoder: {avg_true_decoder_loss:.4f}"
    )
    print(
        f"Full State: {avg_monitoring_full_state_loss:.4f}, Persistence: {avg_persistence_loss:.4f}"
    )

    return avg_loss, total_data_loading_time