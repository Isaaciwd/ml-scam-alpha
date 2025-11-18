"""Training and evaluation utilities for the global atmospheric mixer models."""

from __future__ import annotations

import copy
import gc
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import xarray as xr
from builtins import print as builtin_print
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from VicRegLoss import VICRegLoss
from BarlowTwinsLoss import BarlowTwinsLoss
from CCC_loss import CCCLoss    
try:
    from accelerate import Accelerator  # type: ignore[import]
except ImportError:
    Accelerator = None
import torch._dynamo
torch._dynamo.config.suppress_errors = True


DEFAULT_LOSS_WEIGHTS = {
    'representation': 1.0,
    'true_decoder': 1.0,
    'full_state': 1.0,
    'true_tendency': 2.0,
    'pred_tendency': 4.0,
    'ccc': 4.0,
    'ccc_latent': 1.0,
    'ccc_full_field': 1.0,
}


@dataclass(frozen=True)
class LossWeights:
    """Convenience container for scaling the different loss components."""

    representation: float = DEFAULT_LOSS_WEIGHTS['representation']
    true_decoder: float = DEFAULT_LOSS_WEIGHTS['true_decoder']
    full_state: float = DEFAULT_LOSS_WEIGHTS['full_state']
    true_tendency: float = DEFAULT_LOSS_WEIGHTS['true_tendency']
    pred_tendency: float = DEFAULT_LOSS_WEIGHTS['pred_tendency']
    ccc: float = DEFAULT_LOSS_WEIGHTS['ccc']
    ccc_latent: float = DEFAULT_LOSS_WEIGHTS['ccc_latent']
    ccc_full_field: float = DEFAULT_LOSS_WEIGHTS['ccc_full_field']

    @classmethod
    def from_dict(cls, overrides: Optional[Dict[str, float]]) -> "LossWeights":
        if not overrides:
            return cls()
        params = {**DEFAULT_LOSS_WEIGHTS, **overrides}
        return cls(**params)  # type: ignore[arg-type]

    def as_dict(self) -> Dict[str, float]:
        return {
            'representation': self.representation,
            'true_decoder': self.true_decoder,
            'full_state': self.full_state,
            'true_tendency': self.true_tendency,
            'pred_tendency': self.pred_tendency,
            'ccc': self.ccc,
            'ccc_latent': self.ccc_latent,
            'ccc_full_field': self.ccc_full_field,
        }


def resolve_loss_weights(loss_weights: Optional[Dict[str, float] | LossWeights]) -> LossWeights:
    if isinstance(loss_weights, LossWeights):
        return loss_weights
    return LossWeights.from_dict(loss_weights)


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    *,
    lr: float,
    scheduler_patience: int,
    scheduler_factor: float,
    T_0: int,
    T_mult: int,
    eta_min_cosine: float,
    cosine_lr_t_max_batches: int,
    linear_decay_total_batches: int,
    linear_decay_end_lr: float,
):
    """Create a learning-rate scheduler matching the configuration provided."""

    if scheduler_type == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=0.001,
            threshold_mode='rel',
        )

    if scheduler_type == 'CosineAnnealingWarmRestarts':
        return CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min_cosine)

    if scheduler_type == 'CosineAnnealingLR':
        if cosine_lr_t_max_batches <= 0:
            raise ValueError("cosine_lr_t_max_batches must be positive for CosineAnnealingLR.")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_lr_t_max_batches,
            eta_min=eta_min_cosine,
        )

    if scheduler_type == 'LinearDecayLR':
        if linear_decay_total_batches <= 0:
            raise ValueError("linear_decay_total_batches must be positive for LinearDecayLR.")
        if lr <= 0:
            raise ValueError("Initial learning rate (lr) must be positive for LinearDecayLR.")
        if linear_decay_end_lr < 0:
            raise ValueError("linear_decay_end_lr cannot be negative.")
        if linear_decay_end_lr >= lr:
            builtin_print(
                f"Warning: linear_decay_end_lr ({linear_decay_end_lr}) >= start_lr ({lr}). This will result in LR increasing or staying constant during the decay phase."
            )

        def lr_lambda_linear(current_scheduler_step: int) -> float:
            end_factor = linear_decay_end_lr / lr
            if current_scheduler_step >= linear_decay_total_batches:
                return end_factor
            progress = float(current_scheduler_step) / float(linear_decay_total_batches)
            return 1.0 + (end_factor - 1.0) * progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_linear)
        expected_base_lr = lr
        if any(base_lr != expected_base_lr for base_lr in scheduler.base_lrs):
            scheduler.base_lrs = [expected_base_lr for _ in scheduler.base_lrs]
        return scheduler

    raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")


def accumulate_per_step(
    accumulator: List[float],
    values: List[float],
    weight: float,
) -> List[float]:
    """Accumulate per-step metrics with an optional weighting factor."""

    if not values:
        return accumulator
    if not accumulator:
        return [value * weight for value in values]
    if len(accumulator) != len(values):
        raise ValueError("Per-step metric length mismatch during accumulation.")
    for idx, value in enumerate(values):
        accumulator[idx] += value * weight
    return accumulator


def average_per_step(values: List[float], divisor: float) -> List[float]:
    """Scale accumulated per-step metrics back to averages."""

    if divisor <= 0 or not values:
        return []
    return [value / divisor for value in values]


def print_memory_usage(label: str) -> None:
    """Log host and device memory usage for the current process."""
    process = psutil.Process(os.getpid())
    rss_memory_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[MEM_DEBUG] {label}: RSS Memory = {rss_memory_gb:.3f} GB")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[MEM_DEBUG] {label}: CUDA Memory Allocated = {allocated_gb:.3f} GB, Reserved = {reserved_gb:.3f} GB")

def reshape_for_mixer(
    profile_input: torch.Tensor,
    forcing_input: torch.Tensor,
    profile_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten profile and forcing inputs into the mixer layout expected by the model."""

    batch_size, n_input_vars, n_levels, window_dim, _ = profile_input.shape
    _, n_forcing_vars, _, _ = forcing_input.shape
    _, n_output_vars, _ = profile_target.shape

    profile_input = profile_input.permute(0, 3, 4, 1, 2).reshape(
        batch_size,
        window_dim,
        window_dim,
        n_input_vars * n_levels,
    )

    forcing_input = forcing_input.permute(0, 2, 3, 1).reshape(
        batch_size,
        window_dim,
        window_dim,
        n_forcing_vars,
    )
    input_tensor = torch.cat([profile_input, forcing_input], dim=3)

    target_tensor = profile_target.reshape(batch_size, n_output_vars * n_levels)
    target_tensor = torch.cat([target_tensor, forcing_input[:, 1, 1, :]], dim=1)
    target_tensor = target_tensor[:, None, None, :]

    return input_tensor, target_tensor

def calc_additional_loss(
    full_pred: torch.Tensor,
    reconstructed_output: torch.Tensor,
    input_tensor: torch.Tensor,
    tendency_tensor: torch.Tensor,
    full_output_tensor: torch.Tensor,
    mixer_pred: torch.Tensor,
    encoder_target: torch.Tensor,
    criterion: nn.Module,
    tend_pred_rollout: torch.Tensor,
    tend_true_rollout: torch.Tensor,
    tendency_variances: torch.Tensor,
    loss_weights: Optional[LossWeights | Dict[str, float]] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    float,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Dict[str, List[float]],
    torch.Tensor,
]:
    """Compute the auxiliary losses used to stabilise the training procedure."""

    device = full_pred.device
    dtype = full_pred.dtype
    zero_scalar = torch.zeros((), device=device, dtype=dtype)

    norm_tendency_loss = zero_scalar
    num_steps = full_output_tensor.shape[1] if full_output_tensor.ndim > 1 else 1

    true_decoder_loss_per_step: List[torch.Tensor] = []
    monitoring_full_state_loss_per_step: List[torch.Tensor] = []
    true_tendency_loss_per_step: List[torch.Tensor] = []
    pred_tend_loss_per_step: List[torch.Tensor] = []
    ccc_loss_per_step: List[torch.Tensor] = []
    ccc_latent_loss_per_step: List[torch.Tensor] = []
    ccc_full_field_loss_per_step: List[torch.Tensor] = []

    weights = resolve_loss_weights(loss_weights)
    compute_true_decoder = abs(weights.true_decoder) > 0.0
    compute_full_state = abs(weights.full_state) > 0.0
    compute_true_tendency = abs(weights.true_tendency) > 0.0
    compute_pred_tendency = abs(weights.pred_tendency) > 0.0
    compute_ccc = abs(weights.ccc) > 0.0
    compute_ccc_latent = abs(weights.ccc_latent) > 0.0
    compute_ccc_full_field = abs(weights.ccc_full_field) > 0.0

    use_any_ccc = compute_ccc or compute_ccc_latent or compute_ccc_full_field
    ccc_loss_fn = CCCLoss() if use_any_ccc else None

    prev_full_state = input_tensor
    for step in range(num_steps):
        target_step = full_output_tensor[:, step]

        if compute_true_decoder:
            true_decoder_loss_step = criterion(reconstructed_output[:, step], target_step)
        else:
            true_decoder_loss_step = zero_scalar
        true_decoder_loss_per_step.append(true_decoder_loss_step)

        if compute_full_state:
            monitoring_full_state_loss_step = criterion(full_pred[:, step], target_step)
        else:
            monitoring_full_state_loss_step = zero_scalar
        monitoring_full_state_loss_per_step.append(monitoring_full_state_loss_step)

        tend_step = target_step - prev_full_state

        if compute_true_tendency or compute_pred_tendency:
            if isinstance(criterion, nn.MSELoss):
                reduction = lambda tensor: torch.mean(tensor ** 2, dim=(1, 2))  # noqa: E731
            elif isinstance(criterion, nn.L1Loss):
                reduction = lambda tensor: torch.mean(torch.abs(tensor), dim=(1, 2))  # noqa: E731
            elif isinstance(criterion, CCCLoss):
                reduction = lambda tensor_a, tensor_b=tend_step: criterion(tensor_a, tensor_b)  # noqa: E731
            else:
                raise TypeError(f"Unsupported criterion type for tendency loss: {type(criterion)}")

            if isinstance(criterion, CCCLoss):
                true_tendency_raw = reduction(tend_true_rollout[:, step]) if compute_true_tendency else None
                pred_tendency_raw = reduction(tend_pred_rollout[:, step]) if compute_pred_tendency else None
            else:
                true_tendency_raw = reduction(tend_true_rollout[:, step] - tend_step) if compute_true_tendency else None
                pred_tendency_raw = reduction(tend_pred_rollout[:, step] - tend_step) if compute_pred_tendency else None

            true_tendency_loss_step = (
                (true_tendency_raw / tendency_variances).mean() if compute_true_tendency else zero_scalar
            )
            pred_tend_loss_step = (
                (pred_tendency_raw / tendency_variances).mean() if compute_pred_tendency else zero_scalar
            )
        else:
            true_tendency_loss_step = zero_scalar
            pred_tend_loss_step = zero_scalar

        true_tendency_loss_per_step.append(true_tendency_loss_step)
        pred_tend_loss_per_step.append(pred_tend_loss_step)

        if compute_ccc and ccc_loss_fn is not None:
            pred_tend_for_ccc = tend_pred_rollout[:, step]
            ccc_loss_step = ccc_loss_fn.forward(pred_tend_for_ccc[:, :, :, :120], tend_step[:, :, :, :120])
        else:
            ccc_loss_step = zero_scalar
        ccc_loss_per_step.append(ccc_loss_step)

        if compute_ccc_latent and ccc_loss_fn is not None:
            latent_pred_step = mixer_pred[:, step] if mixer_pred.dim() > 2 else mixer_pred
            latent_target_step = encoder_target[:, step] if encoder_target.dim() > 2 else encoder_target
            ccc_latent_loss_step = ccc_loss_fn.forward(latent_pred_step, latent_target_step)
        else:
            ccc_latent_loss_step = zero_scalar
        ccc_latent_loss_per_step.append(ccc_latent_loss_step)

        if compute_ccc_full_field and ccc_loss_fn is not None:
            ccc_full_field_loss_step = ccc_loss_fn.forward(full_pred[:, step], target_step)
        else:
            ccc_full_field_loss_step = zero_scalar
        ccc_full_field_loss_per_step.append(ccc_full_field_loss_step)

        prev_full_state = full_pred[:, step]

    ccc_loss = torch.stack(ccc_loss_per_step).mean()
    ccc_latent_loss = torch.stack(ccc_latent_loss_per_step).mean()
    ccc_full_field_loss = torch.stack(ccc_full_field_loss_per_step).mean()

    true_decoder_loss = torch.stack(true_decoder_loss_per_step).mean()
    monitoring_full_state_loss = torch.stack(monitoring_full_state_loss_per_step).mean()
    true_tendency_loss = torch.stack(true_tendency_loss_per_step).mean()
    pred_tend_loss = torch.stack(pred_tend_loss_per_step).mean()

    persistence_loss = 0.0003
    persistence_loss_per_step = [persistence_loss] * num_steps

    per_step_losses = {
        'true_decoder_loss': [loss.detach().item() for loss in true_decoder_loss_per_step],
        'monitoring_full_state_loss': [loss.detach().item() for loss in monitoring_full_state_loss_per_step],
        'true_tendency_loss': [loss.detach().item() for loss in true_tendency_loss_per_step],
        'pred_tendency_loss': [loss.detach().item() for loss in pred_tend_loss_per_step],
        'persistence_loss': persistence_loss_per_step,
        'CCC_loss': [loss.detach().item() for loss in ccc_loss_per_step],
        'CCC_latent_loss': [loss.detach().item() for loss in ccc_latent_loss_per_step],
        'CCC_full_field_loss': [loss.detach().item() for loss in ccc_full_field_loss_per_step],
    }

    return (
        true_decoder_loss,
        monitoring_full_state_loss,
        persistence_loss,
        norm_tendency_loss,
        true_tendency_loss,
        pred_tend_loss,
        ccc_latent_loss,
        ccc_full_field_loss,
        per_step_losses,
        ccc_loss,
    )


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
                loss_weights: dict = None,
                use_accelerate: bool = False,
                accelerate_kwargs: dict = None,
                mse_loss_type: str = 'mse',
                tendency_variances_path: Optional[Union[str, Path]] = None):
    """Run the full training loop for the hybrid CNN-transformer model."""


    accelerator = None
    print = builtin_print
    if use_accelerate:
        if Accelerator is None:
            raise ImportError(
                "The 'accelerate' package is required when use_accelerate=True. Install it with `pip install accelerate`."
            )
        accelerate_kwargs = accelerate_kwargs or {}
        accelerator = Accelerator(**accelerate_kwargs)
        print = accelerator.print
        device = accelerator.device
    else:
        accelerate_kwargs = accelerate_kwargs or {}

    def _reduce_sum(value):
        scalar = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
        if use_accelerate:
            tensor = torch.as_tensor(scalar, device=device, dtype=torch.float64)
            return accelerator.reduce(tensor, reduction="sum").item()
        return scalar

    def _reduce_sum_list(values):
        if not values:
            return values
        if use_accelerate:
            tensor = torch.as_tensor(values, device=device, dtype=torch.float64)
            tensor = accelerator.reduce(tensor, reduction="sum")
            return tensor.tolist()
        return values

    def _main_process_only(action):
        if not use_accelerate or accelerator.is_main_process:
            action()

    mixed_precision_dtype: Optional[torch.dtype] = None
    if use_accelerate and accelerator.state.mixed_precision != "no":
        if accelerator.state.mixed_precision == 'bf16':
            mixed_precision_dtype = torch.bfloat16
        elif accelerator.state.mixed_precision == 'fp16':
            mixed_precision_dtype = torch.float16

    def _to_device(tensor: torch.Tensor, *, dtype_override: Optional[torch.dtype] = None) -> torch.Tensor:
        target_dtype = dtype_override if dtype_override is not None else mixed_precision_dtype
        if target_dtype is not None:
            return tensor.to(device=device, dtype=target_dtype, non_blocking=True)
        return tensor.to(device=device, non_blocking=True)


    model_config_params = copy.deepcopy(model_config_params)
    if hasattr(model, 'teacher_forcing_requires_grad'):
        model_config_params['teacher_forcing_requires_grad'] = model.teacher_forcing_requires_grad

    if tendency_variances_path is None:
        raise ValueError("tendency_variances_path must be provided for global training.")
    tendency_variances = torch.load(Path(tendency_variances_path)).to(device)
    tend_var_mean = tendency_variances.mean()
    # append the mean to the end of tendency_variances twice for the sin and cos of lat
    tendency_variances = torch.cat([tendency_variances, tend_var_mean.view(1), tend_var_mean.view(1)], dim=0)
    # if were using MAE instead of MSE take the sqrt of the variances to get std devs
    if mse_loss_type == 'mae':
        tendency_variances = torch.sqrt(tendency_variances)



    if use_torch_compile:
        print("Attempting to compile the model with torch.compile()...")
        try:
            # Ensure model is on the correct device before compilation if it matters for the backend
            compiled_model = torch.compile(model)
            # Reassign to model variable if compilation is successful
            model = compiled_model
            model.to(device) # Ensure model is on the correct device after compile
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Warning: torch.compile() failed: {e}. Proceeding without compilation.")
            # Ensure model is still on the correct device if compilation failed
            model.to(device)

    if hydrostatic_lambda > 0 or lapse_rate_lambda > 0:
        stats_path = "/glade/derecho/scratch/idavis/archive/aqua_planet/atm/hist/solin_filled/stats/"
        print("WARNING - Hydrostatic and lapse rate constraints are not fully implimented yet!")
        print(f"Using stats from {stats_path} for these constraints.")
        means = xr.open_dataset(f"{stats_path}/means.nc")
        stds = xr.open_dataset(f"{stats_path}/stds.nc")
        pressure_levels = means['lev'].values.astype(np.float32)
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
    elif mse_loss_type == 'ccc':
        criterion = CCCLoss()
    else:
        raise ValueError(f"Unknown mse_loss_type: {mse_loss_type}. Must be 'mse', 'mae', or 'ccc'.")

    if loss_weights is None and loaded_checkpoint is not None:
        loss_weights = copy.deepcopy(loaded_checkpoint.get('hyperparams', {}).get('loss_weights'))

    loss_weights_obj = resolve_loss_weights(loss_weights)
    loss_weights_dict = loss_weights_obj.as_dict()

    representation_weight = loss_weights_obj.representation
    true_decoder_weight = loss_weights_obj.true_decoder
    full_state_weight = loss_weights_obj.full_state
    true_tendency_weight = loss_weights_obj.true_tendency
    pred_tendency_weight = loss_weights_obj.pred_tendency
    ccc_weight = loss_weights_obj.ccc
    ccc_latent_weight = loss_weights_obj.ccc_latent
    ccc_full_field_weight = loss_weights_obj.ccc_full_field

    print("Using loss component weights:")
    for key, value in loss_weights_dict.items():
        print(f"  {key}: {value:.4f}")

    
    print(f"Using loss function: {loss_type.upper()}")
    if loss_type == 'barlow':
        print(f"  Barlow Twins lambda: {barlow_lambda}")
    elif loss_type == 'vicreg':
        print(f"  VICReg coefficients - inv: 15.0, var: 25.0, cov: 1.0, gamma: 1.0")

    if warmup_eval_steps > 0: init_lr = warmup_start_lr
    else: init_lr = lr
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay) # Use weight_decay
    # Corrected GradScaler initialization: removed invalid 'device' argument
    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)

    scheduler = build_scheduler(
        optimizer,
        scheduler_type,
        lr=lr,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        T_0=T_0,
        T_mult=T_mult,
        eta_min_cosine=eta_min_cosine,
        cosine_lr_t_max_batches=cosine_lr_t_max_batches,
        linear_decay_total_batches=linear_decay_total_batches,
        linear_decay_end_lr=linear_decay_end_lr,
    )

    scheduler_messages = {
        'ReduceLROnPlateau': f"Using ReduceLROnPlateau scheduler with patience={scheduler_patience}, factor={scheduler_factor}",
        'CosineAnnealingWarmRestarts': (
            f"Using CosineAnnealingWarmRestarts scheduler with T_0={T_0} (batches), T_mult={T_mult}, eta_min={eta_min_cosine}. Stepped per batch after warmup."
        ),
        'CosineAnnealingLR': (
            f"Using CosineAnnealingLR scheduler with T_max={cosine_lr_t_max_batches} (batches), eta_min={eta_min_cosine}. Stepped per batch after warmup."
        ),
        'LinearDecayLR': (
            f"Using LinearDecayLR scheduler: start_lr={lr}, end_lr={linear_decay_end_lr}, total_decay_batches={linear_decay_total_batches}. Stepped per batch after warmup."
        ),
    }
    print(scheduler_messages.get(scheduler_type, f"Using {scheduler_type} scheduler."))

    if use_accelerate:
        prepare_items = [model, optimizer, train_loader]
        has_test_loader = test_loader is not None
        has_scheduler = scheduler is not None
        if has_test_loader:
            prepare_items.append(test_loader)
        if has_scheduler:
            prepare_items.append(scheduler)

        prepared = accelerator.prepare(*prepare_items)
        idx = 0
        model = prepared[idx]; idx += 1
        optimizer = prepared[idx]; idx += 1
        train_loader = prepared[idx]; idx += 1
        if has_test_loader:
            test_loader = prepared[idx]; idx += 1
        if has_scheduler:
            scheduler = prepared[idx]; idx += 1

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
    if use_accelerate:
        best_model_state = accelerator.get_state_dict(model)
    else:
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
                scheduler = build_scheduler(
                    optimizer,
                    scheduler_type,
                    lr=lr,
                    scheduler_patience=scheduler_patience,
                    scheduler_factor=scheduler_factor,
                    T_0=T_0,
                    T_mult=T_mult,
                    eta_min_cosine=eta_min_cosine,
                    cosine_lr_t_max_batches=cosine_lr_t_max_batches,
                    linear_decay_total_batches=linear_decay_total_batches,
                    linear_decay_end_lr=linear_decay_end_lr,
                )

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
    running_train_CCC_loss = 0.0
    running_train_CCC_latent_loss = 0.0
    running_train_CCC_full_field_loss = 0.0
    running_train_samples = 0
    eval_step_counter = start_eval_step
    n_evals_done = start_eval_step // eval_steps if eval_steps > 0 else 0
    warmup_final_step = warmup_eval_steps if warmup_eval_steps > 0 else 0
    warmup_start_step = start_eval_step if (warmup_final_step > 0 and resume_optimizer_use_new_lr_config) else 0
    warmup_span = max(1, warmup_final_step - warmup_start_step) if warmup_final_step > warmup_start_step else 0
    
    # Initialize diagnostic loss tracking (not used in backprop) - now as lists for per-step tracking
    running_train_monitoring_full_state_loss_per_step = []
    running_train_monitoring_tendency_loss = 0.0
    running_train_persistence_loss_per_step = []
    running_train_true_decoder_loss_per_step = []
    running_train_true_tendency_loss_per_step = []
    running_train_pred_tendency_loss_per_step = []
    running_train_CCC_loss_per_step = []
    running_train_CCC_latent_loss_per_step = []
    running_train_CCC_full_field_loss_per_step = []
    
    # Initialize current noise factor (will be updated during training if noise scheduling is used)
    current_noise_factor = noise_factor

    # Initialize timer for the first interval's training part
    interval_start_time = time.time()
    train_data_loading_time_in_interval = 0.0
    last_batch_end_time = time.time() # For data loading timing

    early_stop_triggered = False
    model.train()
    for epoch in tqdm.tqdm(range(start_epoch, epochs), desc="Training Epochs"):
        if early_stop_triggered:
            break
        

        for i, data_chunk in enumerate(train_loader):
            # --- Data Loading Time Measurement ---
            batch_load_start_time = time.time()
            train_data_loading_time_in_interval += batch_load_start_time - last_batch_end_time

            if early_stop_triggered:
                break

            input_tensor = _to_device(data_chunk[0])
            tendency_tensor = _to_device(data_chunk[1])
            full_output_tensor = _to_device(data_chunk[2])

            num_rollout_steps = full_output_tensor.shape[1]


            mixer_pred, encoder_target, full_pred, reconstructed_output, tend_pred_rollout, tend_true_rollout = model(
                input_tensor, full_output_tensor, num_rollout_steps=num_rollout_steps, decode_freq=1, train=True
            )
            # Calculate loss based on selected type
            if loss_type == 'vicreg':
                loss_dict = criterion_vicreg(mixer_pred.view(-1, mixer_pred.shape[-1]), encoder_target.view(-1, encoder_target.shape[-1]))
                loss = loss_dict["loss"]
                variance_loss = loss_dict["var-loss"].item()
                invariance_loss = loss_dict["inv-loss"].item()
                covariance_loss = loss_dict["cov-loss"].item()
            elif loss_type == 'barlow':
                loss = criterion_barlow(mixer_pred.view(-1, mixer_pred.shape[-1]), encoder_target.view(-1, encoder_target.shape[-1]))
                # For Barlow Twins, calculate variance and invariance manually for diagnostics
                with torch.no_grad():
                    # Calculate variance (measure of collapse)
                    # Use unbiased=False and add epsilon for numerical stability
                    mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1])
                    if mixer_pred_flat.size(0) > 1:
                        variance_loss = torch.var(mixer_pred_flat, dim=0, unbiased=False).mean().item()
                    else:
                        variance_loss = 0.0
                    # Calculate invariance (similarity between predictions and targets)
                invariance_loss = criterion(mixer_pred, encoder_target).item()
                covariance_loss = 0.0  # Not directly comparable to VICReg covariance

            else:
                if mse_loss_type in ['mse', 'mae']:
                    MSE_encoded = criterion(mixer_pred, encoder_target)
                else:
                    # loop through time dimension and average CCC loss\
                    MSE_encoded = 0
                    for t in range(mixer_pred.shape[1]):
                        MSE_encoded += criterion(mixer_pred[:, t], encoder_target[:, t])
                    MSE_encoded /= mixer_pred.shape[1]

                loss = MSE_encoded
                variance_loss = 0.0
                invariance_loss = MSE_encoded.item()
                covariance_loss = 0.0
                # dynamically scale so each loss is about equal
                
            true_decoder_loss, monitoring_full_state_loss, persistence_loss, norm_tendency_loss, true_tendency_loss, pred_tend_loss, CCC_latent_loss, CCC_full_field_loss, per_step_losses, CCC_loss = calc_additional_loss(
                full_pred,
                reconstructed_output,
                input_tensor,
                tendency_tensor,
                full_output_tensor,
                mixer_pred,
                encoder_target,
                criterion,
                tend_pred_rollout,
                tend_true_rollout,
                tendency_variances,
                loss_weights_obj,
            )

            true_decoder_loss_val = true_decoder_loss.detach()
            monitoring_full_state_loss_val = monitoring_full_state_loss.detach()
            norm_tendency_loss_val = norm_tendency_loss.detach()
            persistence_loss_val = float(persistence_loss)
            true_tendency_loss_val = true_tendency_loss.detach()
            pred_tend_loss_val = pred_tend_loss.detach()
            CCC_latent_loss_val = CCC_latent_loss.detach()
            CCC_full_field_loss_val = CCC_full_field_loss.detach()
            CCC_loss_val = CCC_loss.detach()

            # normalize losses by their magnitudes
            true_decoder_loss_norm = true_decoder_loss / (true_decoder_loss_val.abs() + 1e-10)
            full_state_norm = monitoring_full_state_loss / (monitoring_full_state_loss_val.abs() + 1e-10)
            true_tend_norm = true_tendency_loss / (true_tendency_loss_val.abs() + 1e-10)
            pred_tend_norm = pred_tend_loss / (pred_tend_loss_val.abs() + 1e-10)
            loss_norm = loss / (loss.detach() + 1e-10)  # Only normalize once!
            CCC_latent_loss_norm = CCC_latent_loss / (CCC_latent_loss_val.abs() + 1e-10)
            CCC_full_field_loss_norm = CCC_full_field_loss / (CCC_full_field_loss_val.abs() + 1e-10)
            CCC_loss_norm = CCC_loss / (CCC_loss_val.abs() + 1e-10)

            representation_loss_val = loss.detach()

            loss_backprop = (
                representation_weight * loss_norm
                + true_decoder_weight * true_decoder_loss_norm
                + full_state_weight * full_state_norm
                + true_tendency_weight * true_tend_norm
                + pred_tendency_weight * pred_tend_norm
                + ccc_latent_weight * CCC_latent_loss_norm
                + ccc_full_field_weight * CCC_full_field_loss_norm
                + ccc_weight * CCC_loss_norm
            )

            weighted_loss_value = (
                representation_weight * representation_loss_val
                + true_decoder_weight * true_decoder_loss_val
                + full_state_weight * monitoring_full_state_loss_val
                + true_tendency_weight * true_tendency_loss_val
                + pred_tendency_weight * pred_tend_loss_val
                + ccc_latent_weight * CCC_latent_loss_val
                + ccc_full_field_weight * CCC_full_field_loss_val
                + ccc_weight * CCC_loss_val
            )


            # Calculate output statistics (same for both loss types)
            # Fix: compute std over the batch dimension, handle edge cases
            mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1])
            if mixer_pred_flat.size(0) > 1:
                output_std = torch.std(mixer_pred_flat, dim=0, unbiased=False).mean().item()
            else:
                output_std = 0.0

            if mixer_pred_flat.numel() > 0:
                output_magnitude = torch.linalg.vector_norm(mixer_pred_flat, dim=1).mean().item()
            else:
                output_magnitude = 0.0
            

            if l1_target_ratio > 0:
                loss_val = loss_backprop.item()
                l1_raw_penalty = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
                if l1_raw_penalty.item() > 1e-10:
                    adaptive_l1_lambda = l1_target_ratio * loss_val / l1_raw_penalty.item()
                    l1_loss = adaptive_l1_lambda * l1_raw_penalty
                    loss_backprop = loss_backprop + l1_loss

            optimizer.zero_grad(set_to_none=True)

            if use_accelerate:
                loss_to_backward = loss_backprop / accelerator.gradient_accumulation_steps
                accelerator.backward(loss_to_backward)
                if grad_clip_max_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                optimizer.step()
            elif use_mixed_precision:
                scaler.scale(loss_backprop).backward()
                scaler.unscale_(optimizer)
                if grad_clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_backprop.backward()
                if grad_clip_max_norm > 0:
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
            running_train_samples += 1
            running_train_CCC_loss += CCC_loss_val
            running_train_CCC_latent_loss += CCC_latent_loss_val
            running_train_CCC_full_field_loss += CCC_full_field_loss_val
            eval_step_counter += 1
            
            # Track diagnostic losses per step (not used in backprop)
            # Accumulate per-step losses
            running_train_monitoring_full_state_loss_per_step = accumulate_per_step(
                running_train_monitoring_full_state_loss_per_step,
                per_step_losses['monitoring_full_state_loss'],
                1.0,
            )
            running_train_persistence_loss_per_step = accumulate_per_step(
                running_train_persistence_loss_per_step,
                per_step_losses['persistence_loss'],
                1.0,
            )
            running_train_true_decoder_loss_per_step = accumulate_per_step(
                running_train_true_decoder_loss_per_step,
                per_step_losses['true_decoder_loss'],
                1.0,
            )
            running_train_true_tendency_loss_per_step = accumulate_per_step(
                running_train_true_tendency_loss_per_step,
                per_step_losses['true_tendency_loss'],
                1.0,
            )
            running_train_pred_tendency_loss_per_step = accumulate_per_step(
                running_train_pred_tendency_loss_per_step,
                per_step_losses['pred_tendency_loss'],
                1.0,
            )
            running_train_CCC_loss_per_step = accumulate_per_step(
                running_train_CCC_loss_per_step,
                per_step_losses['CCC_loss'],
                1.0,
            )
            running_train_CCC_latent_loss_per_step = accumulate_per_step(
                running_train_CCC_latent_loss_per_step,
                per_step_losses['CCC_latent_loss'],
                1.0,
            )
            running_train_CCC_full_field_loss_per_step = accumulate_per_step(
                running_train_CCC_full_field_loss_per_step,
                per_step_losses['CCC_full_field_loss'],
                1.0,
            )

            if eval_step_counter == 3:
                total_samples_preview = _reduce_sum(running_train_samples)
                if total_samples_preview == 0:
                    total_samples_preview = 1.0
                train_loss_preview = _reduce_sum(running_train_loss) / total_samples_preview
                variance_preview = _reduce_sum(running_train_variance) / total_samples_preview
                invariance_preview = _reduce_sum(running_train_invariance) / total_samples_preview
                covariance_preview = _reduce_sum(running_train_covariance) / total_samples_preview
                output_std_preview = _reduce_sum(running_train_output_std) / total_samples_preview
                output_mag_preview = _reduce_sum(running_train_output_magnitude) / total_samples_preview
                norm_tendency_preview = _reduce_sum(running_train_tendency_loss) / total_samples_preview
                true_tendency_preview = _reduce_sum(running_train_true_tendency_loss) / total_samples_preview
                pred_tendency_preview = _reduce_sum(running_train_pred_tendency_loss) / total_samples_preview
                decoder_preview = _reduce_sum(running_train_true_decoder_loss) / total_samples_preview
                ccc_preview = _reduce_sum(running_train_CCC_loss) / total_samples_preview
                ccc_latent_preview = _reduce_sum(running_train_CCC_latent_loss) / total_samples_preview
                ccc_full_field_preview = _reduce_sum(running_train_CCC_full_field_loss) / total_samples_preview

                summed_full_state = _reduce_sum_list(running_train_monitoring_full_state_loss_per_step)
                summed_persistence = _reduce_sum_list(running_train_persistence_loss_per_step)
                summed_ccc = _reduce_sum_list(running_train_CCC_loss_per_step)
                summed_ccc_latent = _reduce_sum_list(running_train_CCC_latent_loss_per_step)
                summed_ccc_full_field = _reduce_sum_list(running_train_CCC_full_field_loss_per_step)

                avg_full_state_per_step = average_per_step(summed_full_state, total_samples_preview)
                avg_persistence_per_step = average_per_step(summed_persistence, total_samples_preview)
                avg_CCC_per_step = average_per_step(summed_ccc, total_samples_preview)
                avg_CCC_latent_per_step = average_per_step(summed_ccc_latent, total_samples_preview)
                avg_CCC_full_field_per_step = average_per_step(summed_ccc_full_field, total_samples_preview)

                print()
                print("Train Loss of first 4 batches:", train_loss_preview)
                if loss_type == 'vicreg':
                    print(f"  Components - Var: {variance_preview:.4f}, Inv: {invariance_preview:.4f}, Cov: {covariance_preview:.4f}")
                elif loss_type == 'barlow':
                    print(f"  Diagnostics - Var: {variance_preview:.4f}, Inv (MSE): {invariance_preview:.4f}")
                print(f"  Output Stats - Std: {output_std_preview:.4f}, Magnitude: {output_mag_preview:.4f}")
                print(f"  Norm Tendency Loss: {norm_tendency_preview:.4f}, True Tendency Loss: {true_tendency_preview:.4f}, Pred Tendency Loss: {pred_tendency_preview:.4f}, Encoder-Decoder Loss: {decoder_preview:.4f}")
                print(f"  CCC Loss: {ccc_preview:.4f}, CCC Latent Loss: {ccc_latent_preview:.4f}")
                print(f"  CCC Full-Field Loss: {ccc_full_field_preview:.7f}")
                print(f"  Diagnostic Losses Per Step:")
                print(f"    Full State: {[f'{loss:.4f}' for loss in avg_full_state_per_step]}")
                print(f"    Persistence: {[f'{loss:.4f}' for loss in avg_persistence_per_step]}")
                print(f"    CCC: {[f'{loss:.4f}' for loss in avg_CCC_per_step]}")
                print(f"    CCC Latent: {[f'{loss:.4f}' for loss in avg_CCC_latent_per_step]}")
                print(f"    CCC Full Field: {[f'{loss:.7f}' for loss in avg_CCC_full_field_per_step]}")
                print()

            warmup_active = warmup_span > 0 and warmup_final_step > 0 and eval_step_counter <= warmup_final_step

            if warmup_active:
                progress_numerator = max(0.0, min(float(warmup_span), float(eval_step_counter - warmup_start_step)))
                progress = progress_numerator / float(warmup_span)
                cosine_factor = 0.5 * (1 - math.cos(progress * math.pi))
                current_lr = warmup_start_lr + (lr - warmup_start_lr) * cosine_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                if isinstance(scheduler, (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                                            torch.optim.lr_scheduler.CosineAnnealingLR,
                                            torch.optim.lr_scheduler.LambdaLR)):
                    scheduler.step()

            # --- End of Batch ---
            # Update timer for data loading measurement
            last_batch_end_time = time.time()

            if eval_step_counter % eval_steps == 0 and running_train_samples > 0:
                # End of training for the current interval, start of evaluation specific timing
                current_time_before_eval = time.time()

                n_evals_done +=1
                total_train_samples = _reduce_sum(running_train_samples)
                current_eval_lr = optimizer.param_groups[0]['lr']
                if total_train_samples > 0:
                    total_train_loss = _reduce_sum(running_train_loss)
                    total_train_variance = _reduce_sum(running_train_variance)
                    total_train_invariance = _reduce_sum(running_train_invariance)
                    total_train_covariance = _reduce_sum(running_train_covariance)
                    total_train_output_std = _reduce_sum(running_train_output_std)
                    total_train_output_magnitude = _reduce_sum(running_train_output_magnitude)
                    total_train_tendency_loss = _reduce_sum(running_train_tendency_loss)
                    total_train_true_tendency_loss = _reduce_sum(running_train_true_tendency_loss)
                    total_train_true_decoder_loss = _reduce_sum(running_train_true_decoder_loss)
                    total_train_pred_tendency_loss = _reduce_sum(running_train_pred_tendency_loss)
                    total_train_CCC_loss = _reduce_sum(running_train_CCC_loss)
                    total_train_CCC_latent_loss = _reduce_sum(running_train_CCC_latent_loss)
                    total_train_CCC_full_field_loss = _reduce_sum(running_train_CCC_full_field_loss)

                    summed_full_state = _reduce_sum_list(running_train_monitoring_full_state_loss_per_step)
                    summed_persistence = _reduce_sum_list(running_train_persistence_loss_per_step)
                    summed_true_decoder = _reduce_sum_list(running_train_true_decoder_loss_per_step)
                    summed_true_tendency = _reduce_sum_list(running_train_true_tendency_loss_per_step)
                    summed_pred_tendency = _reduce_sum_list(running_train_pred_tendency_loss_per_step)
                    summed_ccc = _reduce_sum_list(running_train_CCC_loss_per_step)
                    summed_ccc_latent = _reduce_sum_list(running_train_CCC_latent_loss_per_step)
                    summed_ccc_full_field = _reduce_sum_list(running_train_CCC_full_field_loss_per_step)

                    avg_train_loss = total_train_loss / total_train_samples
                    avg_train_variance = total_train_variance / total_train_samples
                    avg_train_invariance = total_train_invariance / total_train_samples
                    avg_train_covariance = total_train_covariance / total_train_samples
                    avg_train_output_std = total_train_output_std / total_train_samples
                    avg_train_output_magnitude = total_train_output_magnitude / total_train_samples
                    avg_train_tendency_loss = total_train_tendency_loss / total_train_samples
                    avg_train_true_tendency_loss = total_train_true_tendency_loss / total_train_samples
                    avg_train_true_decoder_loss = total_train_true_decoder_loss / total_train_samples
                    avg_train_pred_tendency_loss = total_train_pred_tendency_loss / total_train_samples
                    avg_train_CCC_loss = total_train_CCC_loss / total_train_samples
                    avg_train_CCC_latent_loss = total_train_CCC_latent_loss / total_train_samples
                    avg_train_CCC_full_field_loss = total_train_CCC_full_field_loss / total_train_samples

                    avg_train_monitoring_full_state_loss_per_step = average_per_step(summed_full_state, total_train_samples)
                    avg_train_persistence_loss_per_step = average_per_step(summed_persistence, total_train_samples)
                    avg_train_true_decoder_loss_per_step = average_per_step(summed_true_decoder, total_train_samples)
                    avg_train_true_tendency_loss_per_step = average_per_step(summed_true_tendency, total_train_samples)
                    avg_train_pred_tendency_loss_per_step = average_per_step(summed_pred_tendency, total_train_samples)
                    avg_train_CCC_loss_per_step = average_per_step(summed_ccc, total_train_samples)
                    avg_train_CCC_latent_loss_per_step = average_per_step(summed_ccc_latent, total_train_samples)
                    avg_train_CCC_full_field_loss_per_step = average_per_step(summed_ccc_full_field, total_train_samples)
                else:
                    avg_train_loss = 0.0
                    avg_train_variance = 0.0
                    avg_train_invariance = 0.0
                    avg_train_covariance = 0.0
                    avg_train_output_std = 0.0
                    avg_train_output_magnitude = 0.0
                    avg_train_tendency_loss = 0.0
                    avg_train_true_tendency_loss = 0.0
                    avg_train_true_decoder_loss = 0.0
                    avg_train_pred_tendency_loss = 0.0
                    avg_train_CCC_loss = 0.0
                    avg_train_CCC_latent_loss = 0.0
                    avg_train_CCC_full_field_loss = 0.0
                    avg_train_monitoring_full_state_loss_per_step = []
                    avg_train_persistence_loss_per_step = []
                    avg_train_true_decoder_loss_per_step = []
                    avg_train_true_tendency_loss_per_step = []
                    avg_train_pred_tendency_loss_per_step = []
                    avg_train_CCC_loss_per_step = []
                    avg_train_CCC_latent_loss_per_step = []
                    avg_train_CCC_full_field_loss_per_step = []

                train_losses.append(avg_train_loss)
                lr_history.append(current_eval_lr)
                noise_factor_history.append(current_noise_factor)

                print(f"--- Eval Cycle {n_evals_done} (Epoch {epoch}, Step {eval_step_counter}) ---")
                if total_train_samples > 0:
                    print(f"  Train Loss: {avg_train_loss:.6f}")
                    if loss_type == 'vicreg':
                        print(f"    Components - Var: {avg_train_variance:.4f}, Inv: {avg_train_invariance:.4f}, Cov: {avg_train_covariance:.4f}")
                    elif loss_type == 'barlow':
                        print(f"    Diagnostics - Var: {avg_train_variance:.4f}, Inv (MSE): {avg_train_invariance:.4f}")
                    print(f"    Output Stats - Std: {avg_train_output_std:.4f}, Magnitude: {avg_train_output_magnitude:.4f}")
                    print(
                        f"    calc_additional_loss - Norm Tendency: {avg_train_tendency_loss:.4f}, "
                        f"True Tendency: {avg_train_true_tendency_loss:.4f}, Pred Tendency: {avg_train_pred_tendency_loss:.4f}, Decoder: {avg_train_true_decoder_loss:.4f}, "
                        f"CCC: {avg_train_CCC_loss:.4f}, CCC Latent: {avg_train_CCC_latent_loss:.4f}, CCC Full Field: {avg_train_CCC_full_field_loss:.7f}"
                    )
                    print(f"    Monitoring Losses Per Step:")
                    print(f"      True Decoder: {[f'{loss:.7f}' for loss in avg_train_true_decoder_loss_per_step]}")
                    print(f"      Full State: {[f'{loss:.7f}' for loss in avg_train_monitoring_full_state_loss_per_step]}")
                    print(f"      True Tendency: {[f'{loss:.4f}' for loss in avg_train_true_tendency_loss_per_step]}")
                    print(f"      Pred Tendency: {[f'{loss:.4f}' for loss in avg_train_pred_tendency_loss_per_step]}")
                    print(f"      Persistence: {[f'{loss:.4f}' for loss in avg_train_persistence_loss_per_step]}")
                    print(f"      CCC: {[f'{loss:.7f}' for loss in avg_train_CCC_loss_per_step]}")
                    print(f"      CCC Latent: {[f'{loss:.7f}' for loss in avg_train_CCC_latent_loss_per_step]}")
                    print(f"      CCC Full Field: {[f'{loss:.7f}' for loss in avg_train_CCC_full_field_loss_per_step]}")
                else:
                    print("  Train Loss: N/A (no training batches processed before evaluation)")
                    print("    Monitoring Losses Per Step: []")

                print()

                eval_start_time = time.time()
                avg_test_loss, test_data_loading_time = evaluate_loss(
                    model,
                    test_loader,
                    device,
                    1,
                    use_mixed_precision,
                    loss_type=loss_type,
                    barlow_lambda=barlow_lambda,
                    tendency_variances=tendency_variances,
                    loss_weights=loss_weights,
                    accelerator=accelerator,
                    use_accelerate=use_accelerate,
                )
                evaluation_duration_for_interval = time.time() - eval_start_time
                test_losses.append(avg_test_loss)
                model.train()
                
                print(f"  Test Loss:  {avg_test_loss:.6f}")
                print(f"  LR:         {optimizer.param_groups[0]['lr']:.2e}")
                if noise_factor > 0:
                    print(f"  Noise Factor: {current_noise_factor:.2e}")
                
                if print_interval_timing:
                    training_duration_for_interval = current_time_before_eval - interval_start_time
                    print(f"  Timing: Training part = {training_duration_for_interval:.2f}s (Data Loading: {train_data_loading_time_in_interval:.2f}s)")
                    print(f"          Evaluation part = {evaluation_duration_for_interval:.2f}s (Data Loading: {test_data_loading_time:.2f}s)")
                print(f"--- End Eval Cycle {n_evals_done} ---")

                if use_accelerate:
                    accelerator.wait_for_everyone()

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
                running_train_true_decoder_loss = 0.0
                running_train_pred_tendency_loss = 0.0
                running_train_CCC_loss = 0.0
                running_train_CCC_latent_loss = 0.0
                running_train_CCC_full_field_loss = 0.0
                running_train_samples = 0
                
                # Reset diagnostic loss accumulators (per-step lists)
                running_train_monitoring_full_state_loss_per_step = []
                running_train_monitoring_tendency_loss = 0.0
                running_train_persistence_loss_per_step = []
                running_train_true_decoder_loss_per_step = []
                running_train_true_tendency_loss_per_step = []
                running_train_pred_tendency_loss_per_step = []
                running_train_CCC_loss_per_step = []
                running_train_CCC_latent_loss_per_step = []
                running_train_CCC_full_field_loss_per_step = []

                if not warmup_active and isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_test_loss)

                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    if use_accelerate:
                        best_model_state = accelerator.get_state_dict(model)
                    else:
                        best_model_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    steps_no_improve = 0

                    def _save_best_checkpoint():
                        if not model_path:
                            return
                        checkpoint = {
                            'epoch': epoch,
                            'eval_step': eval_step_counter,
                            'model_state_dict': best_model_state,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
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
                                'loss_weights': copy.deepcopy(loss_weights_dict)
                            }
                        }
                        torch.save(checkpoint, model_path)
                        print(f"New best model saved to {model_path} (Test Loss: {best_test_loss:.6f} at Epoch {best_epoch}, Step {eval_step_counter})")

                    _main_process_only(_save_best_checkpoint)
                else:
                    steps_no_improve += 1
                    print(f"No improvement in test loss for {steps_no_improve} evaluation cycles.")

                if steps_no_improve >= patience:
                    print(f"Early stopping triggered after {patience} evaluation cycles without improvement.")
                    early_stop_triggered = True
                    break
                print()
            if early_stop_triggered:
                break

            # Allow chunk tensors to fall out of scope naturally

        if early_stop_triggered:
            break

    print("Training finished.")
    if best_epoch != -1:
        print(f"Best model had Test Loss: {best_test_loss:.6f} (achieved at Epoch {best_epoch})")
        if use_accelerate:
            accelerator.unwrap_model(model).load_state_dict(best_model_state)
            accelerator.wait_for_everyone()
        else:
            model.load_state_dict(best_model_state)
    else:
        print("No improvement over initial state or training was too short. Returning current model state.")

    return train_losses, test_losses, lr_history, noise_factor_history, model

def evaluate_loss(
    model,
    data_loader,
    device,
    eval_batch_size,
    use_mixed_precision: bool = False,
    max_files_to_evaluate: int = 128,
    loss_type: str = 'vicreg',
    barlow_lambda: float = 5e-3,
    tendency_variances=None,
    loss_weights: dict = None,
    accelerator=None,
    use_accelerate: bool = False,
    mse_loss_type: str = 'mse',
):
    """Evaluate the model on the validation set and report aggregated metrics."""
    if tendency_variances is None:
        raise ValueError("tendency_variances must be provided to evaluate_loss to match training loss computation.")

    loss_weights_obj = resolve_loss_weights(loss_weights)

    print_fn = builtin_print
    if accelerator is not None:
        print_fn = accelerator.print
        device = accelerator.device
        use_accelerate = True
    else:
        use_accelerate = bool(use_accelerate)

    eval_mixed_precision_dtype: Optional[torch.dtype] = None
    if use_accelerate and accelerator is not None and accelerator.state.mixed_precision != "no":
        if accelerator.state.mixed_precision == 'bf16':
            eval_mixed_precision_dtype = torch.bfloat16
        elif accelerator.state.mixed_precision == 'fp16':
            eval_mixed_precision_dtype = torch.float16

    def _eval_to_device(tensor: torch.Tensor) -> torch.Tensor:
        if eval_mixed_precision_dtype is not None:
            return tensor.to(device=device, dtype=eval_mixed_precision_dtype, non_blocking=True)
        return tensor.to(device=device, non_blocking=True)

    def _reduce_eval_sum(value):
        if isinstance(value, torch.Tensor):
            value = float(value.detach().item())
        else:
            value = float(value)
        if use_accelerate and accelerator is not None:
            tensor = torch.tensor(value, device=device, dtype=torch.float64)
            return accelerator.reduce(tensor, reduction="sum").item()
        return value

    def _reduce_eval_list(values):
        if not values:
            return values
        tensor = torch.tensor(values, device=device, dtype=torch.float64)
        if use_accelerate and accelerator is not None:
            tensor = accelerator.reduce(tensor, reduction="sum")
        return tensor.tolist()

    representation_weight = loss_weights_obj.representation
    true_decoder_weight = loss_weights_obj.true_decoder
    full_state_weight = loss_weights_obj.full_state
    true_tendency_weight = loss_weights_obj.true_tendency
    pred_tendency_weight = loss_weights_obj.pred_tendency
    ccc_weight = loss_weights_obj.ccc
    ccc_latent_weight = loss_weights_obj.ccc_latent
    ccc_full_field_weight = loss_weights_obj.ccc_full_field

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
    accumulated_CCC_loss = 0.0
    accumulated_CCC_latent_loss = 0.0
    accumulated_CCC_full_field_loss = 0.0
    accumulated_monitoring_full_state_loss_per_step = []
    accumulated_persistence_loss_per_step = []
    accumulated_true_decoder_loss_per_step = []
    accumulated_true_tendency_loss_per_step = []
    accumulated_pred_tendency_loss_per_step = []
    accumulated_CCC_loss_per_step = []
    accumulated_CCC_latent_loss_per_step = []
    accumulated_CCC_full_field_loss_per_step = []
    
    files_processed_count = 0
    criterion_vicreg_eval = VICRegLoss()
    criterion_barlow_eval = BarlowTwinsLoss(lambda_=barlow_lambda)
    if mse_loss_type == 'mse':
        criterion = nn.MSELoss()
    elif mse_loss_type == 'mae':
        criterion = nn.L1Loss()
    elif mse_loss_type == 'ccc':
        criterion = CCCLoss()
    else:
        raise ValueError(f"Unknown mse_loss_type: {mse_loss_type}. Must be 'mse' or 'mae'.")
    total_data_loading_time = 0.0
    last_batch_end_time = time.time()
    
    with torch.enable_grad():
        for i, data_chunk in enumerate(data_loader):
            batch_load_start_time = time.time()
            total_data_loading_time += batch_load_start_time - last_batch_end_time
            if max_files_to_evaluate > 0 and files_processed_count >= max_files_to_evaluate:
                break
            
            # Keep tensors on CPU and only move per mini-batch to limit peak GPU memory
            input_tensor_cpu = data_chunk[0]
            tendency_tensor_cpu = data_chunk[1]
            full_output_tensor_cpu = data_chunk[2]
            
            num_samples_in_chunk = input_tensor_cpu.size(0)
            if num_samples_in_chunk == 0:
                continue
                
            for start_idx in range(0, num_samples_in_chunk, eval_batch_size):
                end_idx = min(start_idx + eval_batch_size, num_samples_in_chunk)
                
                mini_batch_input = _eval_to_device(input_tensor_cpu[start_idx:end_idx])
                mini_batch_tendency = _eval_to_device(tendency_tensor_cpu[start_idx:end_idx])
                mini_batch_full_output = _eval_to_device(full_output_tensor_cpu[start_idx:end_idx])
                
                current_eval_batch_size = mini_batch_input.size(0)

                if current_eval_batch_size == 0:
                    continue
                    
                num_rollout_steps = mini_batch_full_output.shape[1]

                if torch.isnan(mini_batch_input).any() or torch.isinf(mini_batch_input).any():
                    print_fn("!!! WARNING: NaNs or Infs found in mini_batch_input before model call !!!")
                if torch.isnan(mini_batch_full_output).any() or torch.isinf(mini_batch_full_output).any():
                    print_fn("!!! WARNING: NaNs or Infs found in mini_batch_full_output before model call !!!")

                    # Use the same model call as training loop
                (      
                mixer_pred,
                encoder_target,
                full_pred,
                reconstructed_output,
                tend_pred_rollout,
                tend_true_rollout,
                ) = model(
                mini_batch_input,
                mini_batch_full_output,
                num_rollout_steps=num_rollout_steps,
                decode_freq=1,
                train=True,
                )
                (
                true_decoder_loss,
                monitoring_full_state_loss,
                persistence_loss,
                norm_tendency_loss,
                true_tendency_loss,
                pred_tend_loss,
                CCC_latent_loss,
                CCC_full_field_loss,
                per_step_losses,
                CCC_loss,
                ) = calc_additional_loss(
                full_pred,
                reconstructed_output,
                mini_batch_input,
                mini_batch_tendency,
                mini_batch_full_output,
                mixer_pred,
                encoder_target,
                criterion,
                tend_pred_rollout,
                tend_true_rollout,
                tendency_variances,
                loss_weights_obj,
                )
            
                # Calculate loss based on selected type
                if loss_type == 'vicreg':
                    loss_dict = criterion_vicreg_eval(mixer_pred.view(-1, mixer_pred.shape[-1]), encoder_target.view(-1, encoder_target.shape[-1]))
                    loss = loss_dict["loss"]
                    variance_loss = loss_dict["var-loss"].item()
                    invariance_loss = loss_dict["inv-loss"].item()
                    covariance_loss = loss_dict["cov-loss"].item()
                elif loss_type == 'barlow':
                    loss = criterion_barlow_eval(mixer_pred.view(-1, mixer_pred.shape[-1]), encoder_target.view(-1, encoder_target.shape[-1]))
                    
                    # Debug: Check if Barlow loss is NaN
                    if torch.isnan(loss).any():
                        print_fn(f"DEBUG: Barlow loss is NaN!")
                        print_fn(f"  mixer_pred stats: min={mixer_pred.min().item()}, max={mixer_pred.max().item()}, mean={mixer_pred.mean().item()}")
                        print_fn(f"  encoder_target stats: min={encoder_target.min().item()}, max={encoder_target.max().item()}, mean={encoder_target.mean().item()}")
                        print_fn(f"  mixer_pred has NaN: {torch.isnan(mixer_pred).any().item()}")
                        print_fn(f"  encoder_target has NaN: {torch.isnan(encoder_target).any().item()}")
                    
                    # Calculate diagnostics for Barlow Twins
                    mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1]).detach()
                    if mixer_pred_flat.size(0) > 1:
                        variance_loss = torch.var(mixer_pred_flat, dim=0, unbiased=False).mean().item()
                    else:
                        variance_loss = 0.0
                    invariance_loss = criterion(mixer_pred.detach(), encoder_target.detach()).item()
                    covariance_loss = 0.0

                    MSE_decoded = criterion(full_pred, mini_batch_full_output)
                    
                    # Debug: Check if MSE_decoded is NaN
                    if torch.isnan(MSE_decoded).any():
                        print_fn(f"DEBUG: MSE_decoded is NaN!")
                        print_fn(f"  full_pred has NaN: {torch.isnan(full_pred).any().item()}")
                        print_fn(f"  mini_batch_full_output has NaN: {torch.isnan(mini_batch_full_output).any().item()}")
                    
                    loss += MSE_decoded
                else:

                    if mse_loss_type in ['mse', 'mae']:
                        MSE_encoded = criterion(mixer_pred, encoder_target)
                    else:
                        # loop through time dimension and average CCC loss\
                        MSE_encoded = 0
                        for t in range(mixer_pred.shape[1]):
                            MSE_encoded += criterion(mixer_pred[:, t], encoder_target[:, t])
                        MSE_encoded /= mixer_pred.shape[1]
                    loss = MSE_encoded
                    variance_loss = 0.0
                    invariance_loss = MSE_encoded.item()
                    covariance_loss = 0.0
                
                # Add decoder supervision losses
                true_decoder_loss_val = true_decoder_loss.detach()
                monitoring_full_state_loss_val = monitoring_full_state_loss.detach()
                norm_tendency_loss_val = norm_tendency_loss.detach()
                true_tendency_loss_val = true_tendency_loss.detach()
                pred_tend_loss_val = pred_tend_loss.detach()
                CCC_latent_loss_val = CCC_latent_loss.detach()
                CCC_full_field_loss_val = CCC_full_field_loss.detach()
                CCC_loss_val = CCC_loss.detach()
                loss_val = loss.detach()

                loss = (
                    representation_weight * loss_val
                    + true_decoder_weight * true_decoder_loss_val
                    + full_state_weight * monitoring_full_state_loss_val
                    + true_tendency_weight * true_tendency_loss_val
                    + pred_tendency_weight * pred_tend_loss_val
                    + ccc_latent_weight * CCC_latent_loss_val
                    + ccc_full_field_weight * CCC_full_field_loss_val
                    + ccc_weight * CCC_loss_val
                )

                # Debug: Check for NaN in individual components
                if torch.isnan(loss_val).any():
                    print_fn(f"DEBUG: loss_val is NaN")
                if torch.isnan(true_decoder_loss_val).any():
                    print_fn(f"DEBUG: true_decoder_loss_val is NaN")
                if torch.isnan(monitoring_full_state_loss_val).any():
                    print_fn(f"DEBUG: monitoring_full_state_loss_val is NaN")
                if torch.isnan(norm_tendency_loss_val).any():
                    print_fn(f"DEBUG: norm_tendency_loss_val is NaN")
                if torch.isnan(true_tendency_loss_val).any():
                    print_fn(f"DEBUG: true_tendency_loss_val is NaN")
                if torch.isnan(pred_tend_loss_val).any():
                    print_fn(f"DEBUG: pred_tend_loss_val is NaN")
                if torch.isnan(loss).any():
                    print_fn(f"DEBUG: Final loss is NaN after summation")
                    print_fn(f"  loss_val={loss_val.item()}, true_decoder={true_decoder_loss_val.item()}")
                    print_fn(f"  monitoring={monitoring_full_state_loss_val.item()}, true_tend={true_tendency_loss_val.item()}")
                    print_fn(f"  pred_tend={pred_tend_loss_val.item()}")

                # Calculate output statistics
                mixer_pred_flat = mixer_pred.view(-1, mixer_pred.shape[-1])
                if mixer_pred_flat.size(0) > 1:
                    output_std = torch.std(mixer_pred_flat, dim=0, unbiased=False).mean().item()
                else:
                    output_std = 0.0
                output_magnitude = torch.norm(mixer_pred_flat, dim=1).mean().item()
                
                # Accumulate all metrics
                accumulated_variance += variance_loss * current_eval_batch_size
                accumulated_invariance += invariance_loss * current_eval_batch_size
                accumulated_covariance += covariance_loss * current_eval_batch_size
                accumulated_output_std += output_std * current_eval_batch_size
                accumulated_output_magnitude += output_magnitude * current_eval_batch_size

                # Convert tensors to scalars before accumulation
                accumulated_norm_tendency_loss += norm_tendency_loss_val.item() * current_eval_batch_size
                accumulated_true_tendency_loss += true_tendency_loss_val.item() * current_eval_batch_size
                accumulated_pred_tendency_loss += pred_tend_loss_val.item() * current_eval_batch_size
                accumulated_true_decoder_loss += true_decoder_loss_val.item() * current_eval_batch_size
                accumulated_CCC_loss += CCC_loss_val.item() * current_eval_batch_size
                accumulated_CCC_latent_loss += CCC_latent_loss_val.item() * current_eval_batch_size
                accumulated_CCC_full_field_loss += CCC_full_field_loss_val.item() * current_eval_batch_size
                
                # Accumulate per-step losses
                accumulated_monitoring_full_state_loss_per_step = accumulate_per_step(
                    accumulated_monitoring_full_state_loss_per_step,
                    per_step_losses['monitoring_full_state_loss'],
                    float(current_eval_batch_size),
                )
                accumulated_persistence_loss_per_step = accumulate_per_step(
                    accumulated_persistence_loss_per_step,
                    per_step_losses['persistence_loss'],
                    float(current_eval_batch_size),
                )
                accumulated_true_decoder_loss_per_step = accumulate_per_step(
                    accumulated_true_decoder_loss_per_step,
                    per_step_losses['true_decoder_loss'],
                    float(current_eval_batch_size),
                )
                accumulated_true_tendency_loss_per_step = accumulate_per_step(
                    accumulated_true_tendency_loss_per_step,
                    per_step_losses['true_tendency_loss'],
                    float(current_eval_batch_size),
                )
                accumulated_pred_tendency_loss_per_step = accumulate_per_step(
                    accumulated_pred_tendency_loss_per_step,
                    per_step_losses['pred_tendency_loss'],
                    float(current_eval_batch_size),
                )
                accumulated_CCC_loss_per_step = accumulate_per_step(
                    accumulated_CCC_loss_per_step,
                    per_step_losses['CCC_loss'],
                    float(current_eval_batch_size),
                )
                accumulated_CCC_latent_loss_per_step = accumulate_per_step(
                    accumulated_CCC_latent_loss_per_step,
                    per_step_losses['CCC_latent_loss'],
                    float(current_eval_batch_size),
                )
                accumulated_CCC_full_field_loss_per_step = accumulate_per_step(
                    accumulated_CCC_full_field_loss_per_step,
                    per_step_losses['CCC_full_field_loss'],
                    float(current_eval_batch_size),
                )
                
                # Convert tensor to scalar before accumulation
                accumulated_loss += loss.item() * current_eval_batch_size
                accumulated_samples += current_eval_batch_size

            # Let intermediate tensors go out of scope without forcing garbage collection
            files_processed_count += 1
            last_batch_end_time = time.time()
    
    total_samples = _reduce_eval_sum(accumulated_samples)
    total_data_loading_time = _reduce_eval_sum(total_data_loading_time)
    total_loss = _reduce_eval_sum(accumulated_loss)
    total_variance = _reduce_eval_sum(accumulated_variance)
    total_invariance = _reduce_eval_sum(accumulated_invariance)
    total_covariance = _reduce_eval_sum(accumulated_covariance)
    total_output_std = _reduce_eval_sum(accumulated_output_std)
    total_output_magnitude = _reduce_eval_sum(accumulated_output_magnitude)
    total_norm_tendency_loss = _reduce_eval_sum(accumulated_norm_tendency_loss)
    total_true_tendency_loss = _reduce_eval_sum(accumulated_true_tendency_loss)
    total_pred_tendency_loss = _reduce_eval_sum(accumulated_pred_tendency_loss)
    total_true_decoder_loss = _reduce_eval_sum(accumulated_true_decoder_loss)
    total_CCC_loss = _reduce_eval_sum(accumulated_CCC_loss)
    total_CCC_latent_loss = _reduce_eval_sum(accumulated_CCC_latent_loss)
    total_CCC_full_field_loss = _reduce_eval_sum(accumulated_CCC_full_field_loss)
    total_monitoring_full_state = _reduce_eval_list(accumulated_monitoring_full_state_loss_per_step)
    total_persistence = _reduce_eval_list(accumulated_persistence_loss_per_step)
    total_true_decoder_per_step = _reduce_eval_list(accumulated_true_decoder_loss_per_step)
    total_true_tendency_per_step = _reduce_eval_list(accumulated_true_tendency_loss_per_step)
    total_pred_tendency_per_step = _reduce_eval_list(accumulated_pred_tendency_loss_per_step)
    total_CCC_per_step = _reduce_eval_list(accumulated_CCC_loss_per_step)
    total_CCC_latent_per_step = _reduce_eval_list(accumulated_CCC_latent_loss_per_step)
    total_CCC_full_field_per_step = _reduce_eval_list(accumulated_CCC_full_field_loss_per_step)

    if total_samples > 0:
        print_fn(f"DEBUG: accumulated_loss={total_loss}, accumulated_samples={total_samples}")
        avg_loss = total_loss / total_samples
        print_fn(f"DEBUG: avg_loss after division = {avg_loss}")
        avg_variance = total_variance / total_samples
        avg_invariance = total_invariance / total_samples
        avg_covariance = total_covariance / total_samples
        avg_output_std = total_output_std / total_samples
        avg_output_magnitude = total_output_magnitude / total_samples

        avg_norm_tendency_loss = total_norm_tendency_loss / total_samples
        avg_true_tendency_loss = total_true_tendency_loss / total_samples
        avg_pred_tendency_loss = total_pred_tendency_loss / total_samples
        avg_true_decoder_loss = total_true_decoder_loss / total_samples
        avg_CCC_loss = total_CCC_loss / total_samples
        avg_CCC_latent_loss = total_CCC_latent_loss / total_samples
        avg_CCC_full_field_loss = total_CCC_full_field_loss / total_samples

        avg_monitoring_full_state_loss_per_step = average_per_step(total_monitoring_full_state, total_samples)
        avg_persistence_loss_per_step = average_per_step(total_persistence, total_samples)
        avg_true_decoder_loss_per_step = average_per_step(total_true_decoder_per_step, total_samples)
        avg_true_tendency_loss_per_step = average_per_step(total_true_tendency_per_step, total_samples)
        avg_pred_tendency_loss_per_step = average_per_step(total_pred_tendency_per_step, total_samples)
        avg_CCC_loss_per_step = average_per_step(total_CCC_per_step, total_samples)
        avg_CCC_latent_loss_per_step = average_per_step(total_CCC_latent_per_step, total_samples)
        avg_CCC_full_field_loss_per_step = average_per_step(total_CCC_full_field_per_step, total_samples)
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
        avg_CCC_loss = 0.0
        avg_CCC_latent_loss = 0.0
        avg_CCC_full_field_loss = 0.0
        avg_monitoring_full_state_loss_per_step = []
        avg_persistence_loss_per_step = []
        avg_true_decoder_loss_per_step = []
        avg_true_tendency_loss_per_step = []
        avg_pred_tendency_loss_per_step = []
        avg_CCC_loss_per_step = []
        avg_CCC_latent_loss_per_step = []
        avg_CCC_full_field_loss_per_step = []
    
    if loss_type == 'vicreg':
        print_fn(f"    Test Loss Components - Var: {avg_variance:.4f}, Inv: {avg_invariance:.4f}, Cov: {avg_covariance:.4f}")
    elif loss_type == 'barlow':
        print_fn(f"    Test Loss Diagnostics - Var: {avg_variance:.4f}, Inv (MSE): {avg_invariance:.4f}")
    print_fn(f"    Test Output Stats - Std: {avg_output_std:.4f}, Magnitude: {avg_output_magnitude:.4f}")
    print_fn(
        f"    Test calc_additional_loss - Norm Tendency: {avg_norm_tendency_loss:.4f}, "
        f"True Tendency: {avg_true_tendency_loss:.4f}, Pred Tendency: {avg_pred_tendency_loss:.4f}, Decoder: {avg_true_decoder_loss:.4f}, "
        f"CCC: {avg_CCC_loss:.4f}, CCC Latent: {avg_CCC_latent_loss:.4f}, CCC Full Field: {avg_CCC_full_field_loss:.7f}"
    )
    print_fn(f"    Test Monitoring Losses Per Step:")
    if avg_monitoring_full_state_loss_per_step:
        print_fn(f"      True Decoder: {[f'{loss:.7f}' for loss in avg_true_decoder_loss_per_step]}")
        print_fn(f"      Full State: {[f'{loss:.7f}' for loss in avg_monitoring_full_state_loss_per_step]}")
        print_fn(f"      True Tendency: {[f'{loss:.4f}' for loss in avg_true_tendency_loss_per_step]}")
        print_fn(f"      Pred Tendency: {[f'{loss:.4f}' for loss in avg_pred_tendency_loss_per_step]}")
        print_fn(f"      Persistence: {[f'{loss:.4f}' for loss in avg_persistence_loss_per_step]}")
        print_fn(f"      CCC: {[f'{loss:.7f}' for loss in avg_CCC_loss_per_step]}")
        print_fn(f"      CCC Latent: {[f'{loss:.7f}' for loss in avg_CCC_latent_loss_per_step]}")
        print_fn(f"      CCC Full Field: {[f'{loss:.7f}' for loss in avg_CCC_full_field_loss_per_step]}")
    else:
        print_fn(f"      (No per-step data available)")
    
    return avg_loss, total_data_loading_time