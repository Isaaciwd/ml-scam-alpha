import xarray as xr
import numpy as np
import torch
import glob
from pathlib import Path
from typing import Sequence, Optional
from datetime import timedelta

# CODE IS IN BETA: it is not ideal to load this directly from netcdf on the fly, but speed is profficient for now.

class TemporalNetCDFDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset to create input-target pairs from a sequence of NetCDF files.
    Each pair is separated by a specified time delta.
    """
    def __init__(
        self,
        ds_path_pattern: str,
        hours_later: float = 3.0,
        variable_list: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
        stats_path: str | Path | None = None
    ) -> None:
        """
        Args:
            ds_path_pattern (str): Glob pattern for the NetCDF files.
            hours_later (float): The time difference in hours between the input and target.
            variable_list (Sequence[str] | None): Variables to extract from each timestep.
            dtype (torch.dtype): Desired dtype of the returned tensors.
        """
        self.files = sorted(glob.glob(ds_path_pattern))
        if not self.files:
            raise ValueError("No files found for the given pattern.")

        self.dtype = dtype
        self.variables = list(variable_list) if variable_list is not None else ["T", "U", "V", "Z3"]

        if stats_path is None:
            raise ValueError("stats_path must be provided for normalization.")
        stats_dir = Path(stats_path)
        if not stats_dir.exists():
            raise FileNotFoundError(f"stats_path does not exist: {stats_dir}")

        # Assuming half-hour timesteps
        self.time_step_delta = int(round(hours_later / 0.5))

        # Open the first file to get metadata and pre-calculate lat features
        with xr.open_dataset(self.files[0]) as ds:
            self.timesteps_per_file = len(ds["time"])

            # Pre-calculate latitude features once (shape: n_lat, n_lon, 2)
            lat = torch.deg2rad(torch.as_tensor(ds.lat.values, dtype=self.dtype))
            lon_count = len(ds.lon)
            lat_reshaped = lat.unsqueeze(1).unsqueeze(2)
            lat_expanded = lat_reshaped.expand(-1, lon_count, -1)
            self.lat_features = torch.cat(
                [torch.sin(lat_expanded), torch.cos(lat_expanded)], dim=2
            )
        self.lat_features = self.lat_features.to(self.dtype)

        # Load normalization statistics (per variable, per level)
        self.means = self._load_stats_array(stats_dir / "means.nc")
        self.stds = self._load_stats_array(stats_dir / "stds.nc")
        self.residual_means = self._load_stats_array(stats_dir / "residual_means.nc")
        self.residual_stds = self._load_stats_array(stats_dir / "residual_stds.nc")
        self.eps = 1e-6

        self.total_timesteps = len(self.files) * self.timesteps_per_file
        self.lat_features = self.lat_features.contiguous()

    def __len__(self):
        """
        The number of possible input/target pairs.
        """
        return self.total_timesteps - self.time_step_delta

    def _load_stats_array(self, file_path: Path) -> np.ndarray:
        """
        Load per-variable, per-level statistics from a NetCDF file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Missing stats file: {file_path}")

        with xr.open_dataset(file_path) as ds:
            stats = []
            for var in self.variables:
                if var not in ds:
                    raise KeyError(f"Variable '{var}' missing from stats file {file_path}")
                arr = ds[var].values
                arr = np.asarray(arr, dtype=np.float32)
                arr = np.squeeze(arr)  # expect shape (levels,) or (levels, ...)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Expected stats for '{var}' to depend only on level; got shape {arr.shape}"
                    )
                stats.append(arr)

        stacked = np.stack(stats, axis=0)  # (variables, levels)
        return stacked

    def _load_time_slice(self, dataset: xr.Dataset, time_idx: int) -> np.ndarray:
        """
        Load the requested variables at a given timestep as a NumPy array.

        Returns a tensor with shape (variables, levels, lat, lon).
        """
        arrays = [dataset[var].isel(time=time_idx).values for var in self.variables]
        stacked = np.stack(arrays, axis=0)  # (variables, levels, lat, lon)
        return stacked.astype(np.float32, copy=False)
    
    def reshape_arrays(self, arrays):
        arrays = np.transpose(arrays, (2, 3, 0, 1))  # (lat, lon, variables, levels)
        arrays = np.reshape(arrays, (arrays.shape[0], arrays.shape[1], -1))
        return arrays

    def __getitem__(self, idx):
        """
        Retrieves an (input, target) pair with feature engineering for the input.

        Args:
            idx (int): The index of the input timestep across all files.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and target tensors.
        """
        # Calculate file and time indices for the input
        input_file_idx = idx // self.timesteps_per_file
        input_time_in_file_idx = idx % self.timesteps_per_file

        # Calculate file and time indices for the target
        target_idx = idx + self.time_step_delta
        target_file_idx = target_idx // self.timesteps_per_file
        target_time_in_file_idx = target_idx % self.timesteps_per_file

        with xr.open_dataset(self.files[input_file_idx]) as input_ds:
            input_raw = self._load_time_slice(input_ds, input_time_in_file_idx)
        with xr.open_dataset(self.files[target_file_idx]) as target_ds:
            target_raw = self._load_time_slice(target_ds, target_time_in_file_idx)

        residual_raw = target_raw - input_raw

        # Normalize inputs and residuals using precomputed statistics
        denom = self.stds[:, :, None, None] + self.eps
        input_norm = (input_raw - self.means[:, :, None, None]) / denom
        target_norm = (target_raw - self.means[:, :, None, None]) / denom

        residual_denom = self.residual_stds[:, :, None, None] + self.eps
        residual_norm = (residual_raw - self.residual_means[:, :, None, None]) / residual_denom

        input_np = self.reshape_arrays(input_norm)
        target_np_residual = self.reshape_arrays(residual_norm)
        target_np = self.reshape_arrays(target_norm)

        input_tensor = torch.as_tensor(input_np, dtype=self.dtype)
        input_tensor = torch.cat([input_tensor, self.lat_features], dim=2)

        target_residual_tensor = torch.as_tensor(target_np_residual, dtype=self.dtype)
        target_tensor = torch.as_tensor(target_np, dtype=self.dtype)
        target_tensor = torch.cat([target_tensor, self.lat_features], dim=2)

        return input_tensor.contiguous(), target_residual_tensor.contiguous(), target_tensor.contiguous()

    def state_dict(self):
        # No internal state to save for this dataset
        return {}

    def load_state_dict(self, state_dict):
        # No internal state to load for this dataset
        pass

class ARTemporalNetCDFDataset(torch.utils.data.Dataset):
    """Create autoregressive input/target pairs from sequential NetCDF files."""

    def __init__(
        self,
        ds_path_pattern: str,
        hours_later: float = 3.0,
        num_ar_steps: int = 1,
        variable_list: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
        stats_path: str | Path | None = None,
    ) -> None:
        """Args:
            ds_path_pattern: Glob pattern for the NetCDF files.
            hours_later: Time difference (in hours) between autoregressive samples.
            num_ar_steps: Number of autoregressive target steps to produce per sample.
            variable_list: Variables to extract from each timestep.
            dtype: Desired dtype of the returned tensors.
            stats_path: Directory containing normalization statistics (means/stds/residuals).
        """
        if num_ar_steps < 1:
            raise ValueError("num_ar_steps must be at least 1.")

        self.files = sorted(glob.glob(ds_path_pattern))
        if not self.files:
            raise ValueError("No files found for the given pattern.")

        self.dtype = dtype
        self.variables = list(variable_list) if variable_list is not None else ["T", "U", "V", "Z3"]

        if stats_path is None:
            raise ValueError("stats_path must be provided for normalization.")
        stats_dir = Path(stats_path)
        if not stats_dir.exists():
            raise FileNotFoundError(f"stats_path does not exist: {stats_dir}")

        # Assuming half-hour timesteps
        self.time_step_delta = int(round(hours_later / 0.5))
        if self.time_step_delta < 1:
            raise ValueError("hours_later results in a zero time_step_delta; ensure it is at least 0.5 hours.")

        self.num_ar_steps = num_ar_steps

        # Open the first file to get metadata and pre-calculate lat features
        with xr.open_dataset(self.files[0]) as ds:
            self.timesteps_per_file = len(ds["time"])

            # Pre-calculate latitude features once (shape: n_lat, n_lon, 2)
            lat = torch.deg2rad(torch.as_tensor(ds.lat.values, dtype=self.dtype))
            lon_count = len(ds.lon)
            lat_reshaped = lat.unsqueeze(1).unsqueeze(2)
            lat_expanded = lat_reshaped.expand(-1, lon_count, -1)
            self.lat_features = torch.cat(
                [torch.sin(lat_expanded), torch.cos(lat_expanded)], dim=2
            )
        self.lat_features = self.lat_features.to(self.dtype)

        # Load normalization statistics (per variable, per level)
        self.means = self._load_stats_array(stats_dir / "means.nc")
        self.stds = self._load_stats_array(stats_dir / "stds.nc")
        self.residual_means = self._load_stats_array(stats_dir / "residual_means.nc")
        self.residual_stds = self._load_stats_array(stats_dir / "residual_stds.nc")
        self.eps = 1e-6

        self.total_timesteps = len(self.files) * self.timesteps_per_file
        self.lat_features = self.lat_features.contiguous()

    def __len__(self) -> int:
        """The number of possible input/target tuples."""
        available = self.total_timesteps - (self.num_ar_steps * self.time_step_delta)
        return max(0, available)

    def _load_stats_array(self, file_path: Path) -> np.ndarray:
        """Load per-variable, per-level statistics from a NetCDF file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Missing stats file: {file_path}")

        with xr.open_dataset(file_path) as ds:
            stats = []
            for var in self.variables:
                if var not in ds:
                    raise KeyError(f"Variable '{var}' missing from stats file {file_path}")
                arr = ds[var].values
                arr = np.asarray(arr, dtype=np.float32)
                arr = np.squeeze(arr)  # expect shape (levels,) or (levels, ...)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Expected stats for '{var}' to depend only on level; got shape {arr.shape}"
                    )
                stats.append(arr)

        stacked = np.stack(stats, axis=0)  # (variables, levels)
        return stacked

    def _load_time_slice(self, dataset: xr.Dataset, time_idx: int) -> np.ndarray:
        """Load selected variables at a given timestep as a NumPy array.

        Returns a tensor with shape (variables, levels, lat, lon).
        """
        arrays = [dataset[var].isel(time=time_idx).values for var in self.variables]
        stacked = np.stack(arrays, axis=0)
        return stacked.astype(np.float32, copy=False)

    @staticmethod
    def reshape_arrays(arrays: np.ndarray) -> np.ndarray:
        arrays = np.transpose(arrays, (2, 3, 0, 1))  # (lat, lon, variables, levels)
        arrays = np.reshape(arrays, (arrays.shape[0], arrays.shape[1], -1))
        return arrays

    def __getitem__(self, idx: int):
        """Retrieve an input tensor and an autoregressive stack of targets."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Dataset index out of range")

        # Base timestep bookkeeping
        input_file_idx = idx // self.timesteps_per_file
        input_time_in_file_idx = idx % self.timesteps_per_file

        with xr.open_dataset(self.files[input_file_idx]) as input_ds:
            input_raw = self._load_time_slice(input_ds, input_time_in_file_idx)

        targets_raw = []
        residuals_raw = []
        prev_state = input_raw

        for step in range(1, self.num_ar_steps + 1):
            target_idx = idx + step * self.time_step_delta
            target_file_idx = target_idx // self.timesteps_per_file
            target_time_in_file_idx = target_idx % self.timesteps_per_file

            with xr.open_dataset(self.files[target_file_idx]) as target_ds:
                target_raw = self._load_time_slice(target_ds, target_time_in_file_idx)

            targets_raw.append(target_raw)
            residuals_raw.append(target_raw - prev_state)
            prev_state = target_raw

        denom = self.stds[:, :, None, None] + self.eps
        input_norm = (input_raw - self.means[:, :, None, None]) / denom

        residual_denom = self.residual_stds[:, :, None, None] + self.eps

        target_norm_list = [
            (target_raw - self.means[:, :, None, None]) / denom for target_raw in targets_raw
        ]
        residual_norm_list = [
            (residual_raw - self.residual_means[:, :, None, None]) / residual_denom
            for residual_raw in residuals_raw
        ]

        input_np = self.reshape_arrays(input_norm)
        targets_np = [self.reshape_arrays(arr) for arr in target_norm_list]
        residuals_np = [self.reshape_arrays(arr) for arr in residual_norm_list]

        input_tensor = torch.as_tensor(input_np, dtype=self.dtype)
        input_tensor = torch.cat([input_tensor, self.lat_features], dim=2)

        lat_features_ar = self.lat_features.unsqueeze(0).expand(self.num_ar_steps, -1, -1, -1)

        target_tensor = torch.stack(
            [torch.as_tensor(arr, dtype=self.dtype) for arr in targets_np], dim=0
        )
        target_tensor = torch.cat([target_tensor, lat_features_ar], dim=3)

        target_residual_tensor = torch.stack(
            [torch.as_tensor(arr, dtype=self.dtype) for arr in residuals_np], dim=0
        )
        target_residual_tensor = torch.cat([target_residual_tensor, lat_features_ar], dim=3)

        return (
            input_tensor.contiguous(),
            target_residual_tensor.contiguous(),
            target_tensor.contiguous(),
        )

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

def denormalize_target_sample_to_xarray(
    target_sample: torch.Tensor,
    stats_path: str | Path,
    example_file: str | Path,
    variables: Sequence[str],
    delta_t_hours: float,
    *,
    latlon_feature_dim: int = 2,
    eps: float = 1e-6,
) -> xr.DataArray:
    """Convert a normalized rollout tensor back into an unnormalized ``xarray`` DataArray.

    Args:
        target_sample: Normalized tensor with shape ``(batch, n_steps, n_lat, n_lon, M)``.
        stats_path: Directory containing ``means.nc`` and ``stds.nc`` used during normalization.
        example_file: NetCDF file that supplies coordinate metadata (lat, lon, lev, time).
        variables: Ordered sequence of variable names matching the flattening order used
            when the tensor was created (typically the ``TemporalNetCDFDataset`` variables).
        delta_t_hours: Time increment (in hours) between successive decoded steps.
        latlon_feature_dim: Number of auxiliary latitude/longitude features concatenated
            onto the last dimension. These features (e.g., sin/cos(lat)) are dropped prior
            to un-normalization. Defaults to 2.
        eps: Numerical epsilon used to avoid division-by-zero issues when reconstructing
            with very small standard deviations. Defaults to ``1e-6``.

    Returns:
        ``xr.DataArray`` with dimensions ``("sample", "time", "lat", "lon", "variable", "lev")``
        containing unnormalized physical values for each requested variable and level.

    Raises:
        ValueError: If the tensor's feature dimension is incompatible with the provided
            statistics or if the variables have differing level counts.
    """

    if target_sample.dim() != 5:
        raise ValueError("target_sample must have shape (batch, n_steps, n_lat, n_lon, M).")

    batch, n_steps, n_lat, n_lon, feature_dim = target_sample.shape
    stats_dir = Path(stats_path)
    means_path = stats_dir / "means.nc"
    stds_path = stats_dir / "stds.nc"

    if delta_t_hours <= 0:
        raise ValueError("delta_t_hours must be positive.")

    delta_seconds = float(delta_t_hours) * 3600.0
    delta_td = timedelta(seconds=delta_seconds)

    if not means_path.exists() or not stds_path.exists():
        raise FileNotFoundError("means.nc and stds.nc must exist inside stats_path.")

    def _build_time_coords(start_time_value: object) -> np.ndarray:
        if isinstance(start_time_value, np.datetime64):
            delta_ns = int(round(delta_seconds * 1e9))
            if delta_ns == 0:
                raise ValueError("delta_t_hours is too small to create unique time coordinates.")
            delta_np = np.timedelta64(delta_ns, "ns")
            offsets = np.arange(n_steps, dtype=np.int64) * delta_np
            return start_time_value + offsets

        try:
            return np.array(
                [start_time_value + step * delta_td for step in range(n_steps)],
                dtype=object,
            )
        except TypeError as exc:
            raise TypeError(
                "Unable to build time coordinates using the example file's time value type."
            ) from exc

    # Load statistics for the requested variables
    with xr.open_dataset(means_path) as means_ds, xr.open_dataset(stds_path) as stds_ds:
        means = []
        stds = []
        level_counts = []
        for var in variables:
            if var not in means_ds or var not in stds_ds:
                raise KeyError(f"Variable '{var}' missing from statistics datasets at {stats_dir}.")

            mean_vals = np.asarray(means_ds[var].values, dtype=np.float32).squeeze()
            std_vals = np.asarray(stds_ds[var].values, dtype=np.float32).squeeze()

            if mean_vals.ndim != 1 or std_vals.ndim != 1:
                raise ValueError(
                    f"Expected 1-D statistics for variable '{var}', got shapes {mean_vals.shape} and {std_vals.shape}."
                )

            level_counts.append(mean_vals.shape[0])
            means.append(mean_vals)
            stds.append(std_vals)

    if len(set(level_counts)) != 1:
        raise ValueError(
            "All variables are expected to share the same number of vertical levels for reconstruction;"
            f" received counts {level_counts}."
        )

    n_levels = level_counts[0]
    expected_feature_dim = n_levels * len(variables)

    # Remove concatenated latitude features if present
    usable_features = expected_feature_dim
    if feature_dim < usable_features:
        raise ValueError(
            f"target_sample feature dimension ({feature_dim}) is smaller than the expected"
            f" {usable_features} derived from statistics."
        )

    if feature_dim > usable_features:
        remaining = feature_dim - usable_features
        if remaining != latlon_feature_dim:
            raise ValueError(
                "Feature dimension includes unexpected auxiliary features."
                f" Expected remainder {latlon_feature_dim}, received {remaining}. "
            )

    sample_np = target_sample.detach().cpu().numpy()
    sample_core = sample_np[..., :usable_features]

    # Reshape into (batch, steps, lat, lon, variables, levels)
    sample_core = sample_core.reshape(batch, n_steps, n_lat, n_lon, len(variables), n_levels)

    # Invert normalization for each variable
    reconstructed = np.empty_like(sample_core, dtype=np.float32)
    for idx, (mean_vals, std_vals) in enumerate(zip(means, stds)):
        mean_broadcast = mean_vals.reshape((1, 1, 1, 1, 1, n_levels))
        std_broadcast = std_vals.reshape((1, 1, 1, 1, 1, n_levels))
        reconstructed[..., idx, :] = sample_core[..., idx, :] * (std_broadcast + eps) + mean_broadcast

    # Load coordinate metadata from the example file
    example_path = Path(example_file)
    if not example_path.exists():
        raise FileNotFoundError(f"Example NetCDF file not found: {example_path}")

    start_time_value = None

    with xr.open_dataset(example_path) as example_ds:
        lat_values = np.asarray(example_ds["lat"].values, dtype=np.float32)
        lon_values = np.asarray(example_ds["lon"].values, dtype=np.float32)
        if lat_values.shape[0] != n_lat or lon_values.shape[0] != n_lon:
            raise ValueError(
                "Latitude/longitude dimensions from example file do not match the target_sample shape."
            )

        if "time" not in example_ds.coords:
            raise ValueError("Example file must contain a 'time' coordinate with at least one entry.")

        time_values = np.asarray(example_ds["time"].values)
        if time_values.size == 0:
            raise ValueError("Example file must contain at least one time coordinate value.")

        start_time_value = time_values[0]

        if "lev" in example_ds.coords and example_ds["lev"].size > 0:
            lev_values = np.asarray(example_ds["lev"].values, dtype=np.float32)
        else:
            lev_values = np.arange(n_levels, dtype=np.float32)

    if start_time_value is None:
        raise RuntimeError("Failed to extract a starting time coordinate from the example file.")

    time_coord = _build_time_coords(start_time_value)

    if lev_values.shape[0] != n_levels:
        raise ValueError(
            f"Vertical level count ({lev_values.shape[0]}) from example file doesn't match expected {n_levels}."
        )

    coords = {
        "sample": np.arange(batch),
        "time": time_coord,
        "lat": lat_values,
        "lon": lon_values,
        "variable": list(variables),
        "lev": lev_values,
    }

    data_array = xr.DataArray(
        reconstructed,
        dims=("sample", "time", "lat", "lon", "variable", "lev"),
        coords=coords,
        name="unnormalized_targets",
    )

    return data_array