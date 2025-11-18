#!/usr/bin/env python3
"""Autoregressive rollout script for latent global atmosphere mixers.

This utility loads a fine-tuned ``ConvLatentGlobalAtmosMixer`` or
``LatentGlobalAtmosMixer`` checkpoint, prepares a normalized initial state from a
NetCDF file, performs an autoregressive rollout, and saves the decoded physical
predictions as a NetCDF file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence, Type
import time

import numpy as np
import torch
import xarray as xr

from GlobalAtmoMixr import LatentGlobalAtmosMixer
from NNorm_GAM import NNorm_LatentGlobalAtmosMixer
from global_data_loader import (
    ARTemporalNetCDFDataset,
    denormalize_target_sample_to_xarray,
)

VARIABLE_LIST: Sequence[str] = ("T", "U", "V", "Z3")
EPS = 1e-6

SUPPORTED_ARCHITECTURES: Dict[str, Type[torch.nn.Module]] = {
    "ConvLatentGlobalAtmosMixer": ConvLatentGlobalAtmosMixer,
    "LatentGlobalAtmosMixer": LatentGlobalAtmosMixer,
    "NNorm_LatentGlobalAtmosMixer": NNorm_LatentGlobalAtmosMixer,
}

ARCHITECTURE_ALIASES = {
    "OptimizedNNormModule": "NNorm_LatentGlobalAtmosMixer",
    "NNormLatentGlobalAtmosMixer": "NNorm_LatentGlobalAtmosMixer",
}


def _validate_prediction_array(
    prediction_da: xr.DataArray, variables: Sequence[str]
) -> xr.DataArray:
    """Validate and preprocess the prediction DataArray."""
    if "sample" in prediction_da.dims:
        if prediction_da.sizes["sample"] != 1:
            raise ValueError("Only single-sample predictions are supported.")
        prediction_da = prediction_da.isel(sample=0, drop=True)

    if "variable" not in prediction_da.dims:
        raise ValueError("Prediction DataArray must have a 'variable' dimension.")

    prediction_vars = set(
        str(v) for v in prediction_da.coords["variable"].to_numpy()
    )
    if missing := [v for v in variables if v not in prediction_vars]:
        raise KeyError(f"Prediction is missing variables: {', '.join(missing)}")

    return prediction_da


def _get_template_metadata(
    template_path: Path, variables: Sequence[str]
) -> tuple[dict, dict, dict]:
    """Extract metadata from the template NetCDF file."""
    with xr.open_dataset(template_path) as ds:
        template_attrs = dict(ds.attrs)
        coord_sources = {
            name: ds.coords[name].copy()
            for name in ("lat", "lon", "lev")
            if name in ds.coords
        }
        var_metadata = {
            var: (ds[var].dims, dict(ds[var].attrs))
            if var in ds.data_vars
            else (("time", "lev", "lat", "lon"), {})
            for var in variables
        }
    return template_attrs, coord_sources, var_metadata


def build_prediction_dataset(
    prediction_da: xr.DataArray,
    template_path: Path,
    variables: Sequence[str],
) -> xr.Dataset:
    """Reconstruct an output dataset that mirrors the template NetCDF structure."""
    prediction_da = _validate_prediction_array(prediction_da, variables)
    template_attrs, coord_sources, var_metadata = _get_template_metadata(
        template_path, variables
    )

    data_vars = {}
    for var in variables:
        desired_dims, attrs = var_metadata[var]
        var_slice = prediction_da.sel(variable=var).squeeze("variable", drop=True)
        var_slice = var_slice.reset_coords("variable", drop=True)

        if missing_dims := [d for d in desired_dims if d not in var_slice.dims]:
            raise ValueError(
                f"Predictions for '{var}' missing dims: {missing_dims}"
            )

        var_slice = var_slice.transpose(*desired_dims)
        if coord_updates := {
            n: c for n, c in coord_sources.items() if n in var_slice.dims
        }:
            var_slice = var_slice.assign_coords(coord_updates)

        data_vars[var] = xr.DataArray(
            data=var_slice.data,
            dims=var_slice.dims,
            coords=var_slice.coords,
            attrs=attrs,
        )

    output_ds = xr.Dataset(data_vars=data_vars, attrs=template_attrs)
    return output_ds.assign_coords(coord_sources)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an autoregressive rollout with a latent global atmosphere mixer."
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Model architecture. If omitted, inferred from checkpoint.",
    )
    parser.add_argument(
        "--num-rollout-steps",
        type=int,
        default=48,
        help="Number of autoregressive steps to perform.",
    )
    parser.add_argument(
        "--decode-freq",
        type=int,
        default=1,
        help="Frequency (in steps) for decoding to physical space.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (*.pth).",
    )
    parser.add_argument(
        "--input-nc-path",
        type=str,
        required=True,
        help="Path to NetCDF file with initial state.",
    )
    parser.add_argument(
        "--output-nc-path",
        type=str,
        required=True,
        help="Path to save the output forecast NetCDF file.",
    )
    parser.add_argument(
        "--stats-path",
        type=str,
        required=True,
        help="Path to the directory with normalization statistics (means.nc, stds.nc).",
    )
    parser.add_argument(
        "--delta-t-hours",
        type=float,
        default=0.5,
        help="Time increment (in hours) between decoded steps.",
    )
    return parser.parse_args()


def load_stats_array(
    stats_dir: Path, variables: Sequence[str], filename: str
) -> np.ndarray:
    """Load per-variable, per-level statistics from a NetCDF file."""
    path = stats_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing statistics file: {path}")

    arrays = []
    with xr.open_dataset(path) as ds:
        for var in variables:
            if var not in ds:
                raise KeyError(f"Variable '{var}' not found in {path}")
            values = np.asarray(ds[var].values, dtype=np.float32).squeeze()
            if values.ndim != 1:
                raise ValueError(
                    f"Expected 1-D stats for '{var}', got shape {values.shape}."
                )
            arrays.append(values)
    return np.stack(arrays, axis=0)


def build_latlon_features(
    latitudes: np.ndarray, lon_count: int, dtype: torch.dtype
) -> torch.Tensor:
    """Create sin/cos latitude features matching dataset preprocessing."""
    lat_tensor = torch.as_tensor(latitudes, dtype=dtype)
    lat_radians = torch.deg2rad(lat_tensor)
    lat_reshaped = lat_radians.unsqueeze(1).unsqueeze(2)
    lat_expanded = lat_reshaped.expand(-1, lon_count, -1)
    return torch.cat([torch.sin(lat_expanded), torch.cos(lat_expanded)], dim=2)


def reshape_input(array: np.ndarray) -> np.ndarray:
    """Mirror ARTemporalNetCDFDataset's reshape to (lat, lon, vars*levels)."""
    return ARTemporalNetCDFDataset.reshape_arrays(array)


def _load_initial_data(
    initial_ds_path: Path, variables: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the first timestep from the NetCDF dataset."""
    if not initial_ds_path.exists():
        raise FileNotFoundError(f"Initial dataset not found: {initial_ds_path}")

    with xr.open_dataset(initial_ds_path) as ds:
        if "time" not in ds.dims and "time" not in ds.coords:
            raise ValueError("Initial dataset must have a 'time' dimension.")

        slices = []
        for var in variables:
            if var not in ds:
                raise KeyError(f"Variable '{var}' missing from {initial_ds_path}")
            data = ds[var].isel(time=0).values.astype(np.float32)
            if data.ndim != 3:
                raise ValueError(f"Var '{var}' must have shape (lev, lat, lon).")
            slices.append(data)

        input_raw = np.stack(slices, axis=0)
        latitudes = ds["lat"].values.astype(np.float32)
        longitudes = ds["lon"].values.astype(np.float32)

    return input_raw, latitudes, longitudes


def _normalize_data(
    input_raw: np.ndarray, means: np.ndarray, stds: np.ndarray
) -> np.ndarray:
    """Normalize the raw input data."""
    denom = stds[:, :, None, None] + EPS
    return (input_raw - means[:, :, None, None]) / denom


def prepare_initial_input(
    initial_ds_path: Path,
    means: np.ndarray,
    stds: np.ndarray,
    variables: Sequence[str],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Load and prepare the initial input tensor for the model."""
    input_raw, latitudes, longitudes = _load_initial_data(initial_ds_path, variables)
    input_normalized = _normalize_data(input_raw, means, stds)

    reshaped = reshape_input(input_normalized)
    lat_features = build_latlon_features(latitudes, len(longitudes), dtype=dtype)

    if lat_features.shape[:2] != reshaped.shape[:2]:
        raise ValueError("Latitude feature shape mismatch with reshaped input.")

    combined = np.concatenate([reshaped, lat_features.numpy()], axis=2)
    return torch.as_tensor(combined, dtype=dtype)


def infer_architecture_from_state_dict(
    state_dict: Mapping[str, torch.Tensor]
) -> str | None:
    """Infer model architecture by inspecting unique parameter names."""
    keys = state_dict.keys()
    if any("output_conv" in key for key in keys):
        return "ConvLatentGlobalAtmosMixer"
    if any(".norm1." in key or ".norm2." in key for key in keys):
        return "LatentGlobalAtmosMixer"
    if any("center_only_heads" in key for key in keys):
        return "NNorm_LatentGlobalAtmosMixer"
    return None


def _resolve_architecture_name(
    label: str | None, state_dict: Mapping[str, torch.Tensor] | None = None
) -> str | None:
    """Map a stored architecture label to a supported class key."""
    if label is None:
        return None

    resolved = ARCHITECTURE_ALIASES.get(label, label)
    if resolved == "OptimizedModule" and state_dict is not None:
        inferred = infer_architecture_from_state_dict(state_dict)
        if inferred:
            print(
                f"Inferred '{inferred}' from state_dict for compiled checkpoint."
            )
            return inferred

    if resolved not in SUPPORTED_ARCHITECTURES and state_dict is not None:
        inferred = infer_architecture_from_state_dict(state_dict)
        if inferred:
            print(
                f"Label '{resolved}' not recognized. Using inferred '{inferred}'."
            )
            return inferred

    return resolved


def _get_architecture(
    requested_arch: str | None,
    hyperparams: dict,
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """Determine the definitive model architecture to use."""
    saved_arch = hyperparams.get("architecture")
    resolved_saved_arch = _resolve_architecture_name(saved_arch, state_dict)
    inferred_arch = infer_architecture_from_state_dict(state_dict)

    if requested_arch:
        resolved_req_arch = _resolve_architecture_name(requested_arch, state_dict)
        if resolved_req_arch and resolved_saved_arch and resolved_req_arch != resolved_saved_arch:
            print(
                f"Warning: Requested arch '{resolved_req_arch}' differs from "
                f"checkpoint's '{resolved_saved_arch}'. Using requested arch."
            )
        return resolved_req_arch or inferred_arch or "LatentGlobalAtmosMixer"

    if resolved_saved_arch:
        return resolved_saved_arch

    if inferred_arch:
        print(
            f"No architecture in metadata; inferring '{inferred_arch}' from params."
        )
        return inferred_arch

    raise ValueError(
        "Cannot determine model architecture. Please provide --architecture."
    )


def _create_model_from_checkpoint(
    checkpoint: dict, device: torch.device, architecture: str | None
) -> torch.nn.Module:
    """Create and load a model from a checkpoint dictionary."""
    state_dict = checkpoint["model_state_dict"]
    hyperparams = checkpoint.get("hyperparams", {})
    model_config = hyperparams.get("model_config_params")

    if not model_config:
        raise KeyError("Checkpoint missing 'hyperparams[\"model_config_params\"]'.")

    # Clean state_dict keys
    for prefix in ("_orig_mod.", "module."):
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}

    arch_to_use = _get_architecture(architecture, hyperparams, state_dict)
    model_cls = SUPPORTED_ARCHITECTURES.get(arch_to_use)

    if not model_cls:
        raise ValueError(f"Unsupported architecture: '{arch_to_use}'.")

    model = model_cls(**model_config)
    load_result = model.load_state_dict(state_dict, strict=False)

    if load_result.unexpected_keys:
        print(f"Warning: Unexpected params: {', '.join(load_result.unexpected_keys)}")
    if load_result.missing_keys:
        if any(not k.startswith("tend_decoder.") for k in load_result.missing_keys):
            raise ValueError(f"Missing required params: {', '.join(load_result.missing_keys)}")
        print("Warning: tend_decoder weights not found; using random init.")

    model.to(device)
    model.eval()
    print(f"Loaded model '{arch_to_use}' from state dictionary.")
    return model


def load_model(
    model_path: Path,
    device: torch.device,
    *,
    architecture: str | None = None,
) -> torch.nn.Module:
    """Instantiate and load a latent mixer model from a checkpoint."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint._orig_mod if hasattr(checkpoint, "_orig_mod") else checkpoint
        model.to(device)
        model.eval()
        print("Loaded model directly from checkpoint object.")
        return model

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return _create_model_from_checkpoint(checkpoint, device, architecture)

    raise ValueError("Unrecognized checkpoint format.")


def main() -> int:
    """Main entry point for the script."""
    args = parse_args()

    if args.num_rollout_steps < 1:
        raise ValueError("--num-rollout-steps must be at least 1.")
    if args.decode_freq < 1:
        raise ValueError("--decode-freq must be at least 1.")
    if args.delta_t_hours <= 0:
        raise ValueError("--delta-t-hours must be positive.")

    model_path = Path(args.checkpoint_path)
    initial_ds_path = Path(args.input_nc_path)
    stats_dir = Path(args.stats_path)
    output_path = Path(args.output_nc_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device, architecture=args.architecture)

    means = load_stats_array(stats_dir, VARIABLE_LIST, "means.nc")
    stds = load_stats_array(stats_dir, VARIABLE_LIST, "stds.nc")

    input_tensor = prepare_initial_input(
        initial_ds_path, means, stds, VARIABLE_LIST
    )
    if input_tensor.shape[-1] != model.M_features:
        raise ValueError(
            f"Input feature dim ({input_tensor.shape[-1]}) does not match "
            f"model expectation ({model.M_features})."
        )

    input_tensor_dev = input_tensor.unsqueeze(0).contiguous().to(device)

    predictions = run_rollout(
        model, input_tensor_dev, args.num_rollout_steps, args.decode_freq
    )

    save_predictions(
        predictions,
        input_tensor,
        output_path,
        stats_dir,
        initial_ds_path,
        VARIABLE_LIST,
        args.delta_t_hours,
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
