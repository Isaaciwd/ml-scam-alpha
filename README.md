# ALPHA DISCLAIMER

**This project is in an ALPHA stage. Many features may not work as expected, not all features are properly documented, models and data pipelines are not fully optimized, and scripts may not run as expected. Use with caution.**

# ML-SCAM

ML-SCAM: Machine Learning Single Column Atmospheric Model

ML-SCAM is a prototype atmospheric emulator framework that evolves the atmosphere using only local information, and entirely within a learned latent space.

Unlike global machine learning weather models (e.g., GraphCast, PanguWeather, FourCastNet) that operate on the entire global field at once, ML-SCAM takes a localized approach. Each column’s next state depends only on itself and its 8 neighboring columns (a 3×3 local tile), allowing the model to learn physically meaningful interactions such as vertical mixing and horizontal advection without requiring global context. This design is heavily motivated by finite volume dynamical cores, and it enables massive parallelism while dramatically reducing the required model size. It also greatly increases the effective amount of training data, because each grid column provides a separate training example, whereas global models yield only one example per global timestep. Finally, by learning dynamics in a location-independent, local manner, the model is forced to capture physical relationships that generalize across the globe, limiting learning of spurious correlations and potentially improving its ability to generalize to unseen climates. At the same time, this design makes training far more computationally efficient than current SOTA global ML weather and climate models.

Additionally, unlike the current crop of SOTA ML weather/climate models which encode a whole earth state vector, process it, decode, and repeat for autoregressive rollouts, ML-SCAM performs autoregressive rollout entirely in the latent space without decoding. Each vertical column of the atmosphere is individually encoded into a latent vector that captures its thermodynamic and dynamic state. A latent dynamics model (currently an MLP-mixer) then performs autoregressive time stepping in this latent space, predicting the temporal evolution of the encoded atmosphere. The decoder maps latent states back to physical-space variables (e.g., temperature, humidity, winds, surface fluxes) for diagnostics and training loss computation, but decoded states are never fed back into the autoregressive loop — all temporal prediction happens purely in the latent space. This design isolates the learned atmospheric dynamics from reconstruction noise, and encourages smooth latent manifolds known to be beneficial for modeling of high-dimensional physical systems.

This design is especially appropriate for ML-SCAM for two reasons. First, the model’s temporal resolution is constrained by the CFL condition and its intentionally small spatial field of view. Short autoregressive steps mean there is little value in decoding at every intermediate timestep; we only need decoded physical states when producing user-facing forecasts or diagnostic outputs. Skipping unnecessary decodes reduces memory traffic and compute cost during long rollouts. Second, ML-SCAM expands into a higher-dimensional latent space when encoding. Decoding and then re-encoding each step would act as a repeated lossy compression cycle, injecting reconstruction noise back into the dynamical core and potentially degrading long-horizon stability. In contrast, current global ML weather models start from very large state vectors and must compress; decoding there restores spatial detail and can be useful every step. For a lightweight, locality-based latent dynamics model like ML-SCAM, remaining in latent space preserves signal fidelity and efficiency.

### Concordance Correlation Coefficient (CCC) Loss

Additonally, we introduce concordance correlation coefficient (CCC) as a new loss for ML weather/climate models. CCC jointly penalizes mean bias, variance mismatch, and lack of correlation, rewarding predictions that reproduce both the amplitude and temporal/spatial co-variability of the true atmospheric evolution. This is especially valuable for retaining extremes, and sub synoptic scale variability that can be muted under pure MSE/MAE objectives. Standard pointwise losses often encourage overly smooth fields—reducing residual spikes at the cost of damping physically meaningful variability—while CCC keeps variance and covariance fidelity in play, preserving sharp gradients (fronts, convective towers) and realistic power spectra. 

The coefficient is computed as

$$
\rho_c = \frac{2\,\sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}
$$

where $\sigma_{xy}$ is the covariance between predictions and targets, $\sigma_x^2$ and $\sigma_y^2$ are their variances, and $\mu_x$, $\mu_y$ are their means. Maximizing $\rho_c$ encourages the model to match both the amplitude and the distributional spread of the true atmospheric state while maintaining high correlation. $rho_c$ varies between 1 (best fit) and -1 (worst fit), so loss is calculated as $1 - \rho_c$.


## Project Structure

*   `GlobalAtmoMixr.py`, `NNorm_GAM.py` : Core model implementations.
*   `train.py`: Script for training the models.
*   `run_conv_mixer_rollout.py`: Script for running inference and generating rollouts.
*   `global_data_loader.py`: Data loading and preprocessing utilities for NetCDF files.
*   `torch_funcs_global.py`, `offline_torch_funcs_latent.py`: Helper functions for the training and evaluation loops.
*   `BarlowTwinsLoss.py`, `CCC_loss.py`, `VicRegLoss.py`: Custom loss functions.
*   `config.py`: Configuration file for training and inference.
*   `train_offline_latent.py`: Entry point for single-column training runs.
*   `requirements.txt`: Python dependencies for the alpha release.



## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Training and inference parameters are managed via `config.py`. You can create multiple configuration files (e.g., `config_gpu.py`, `config_cpu.py`) and specify which one to use with the `--config` argument.

### Training

To train a model, use the `train.py` script. By default, it uses `config.py`. You can specify a different configuration file:

```bash
python train.py --config my_custom_config
```

Adjust data paths and other parameters in your chosen configuration file.

Training can be done in single column mode, global mode, or in global autoregressive rollouts, depending on the settings in your configuration file. Single column mode feeds in batches of random 3x3 patches of grid columns at time T, training the model to make an accurate prediction of the center column at time T+1. Global mode feeds in the entire global state at time T, training the model to predict the entire global state at time T+1, though it still only makes forecasts one column at a time. Global autoregressive rollouts feed in the entire global state at time T and train the model to make multi-timestep forecasts of the entire global state, feeding its own predictions back in as input for subsequent timesteps. Scripts to create single column training data are not yet provided. In global training the model  trains directly from global NetCDF files containing the full atmospheric state.

### Inference

To run a rollout with a trained model, use the `run_conv_mixer_rollout.py` script. It also uses `config.py` by default, but you can override parameters via command-line arguments or a custom config file:

```bash
python run_conv_mixer_rollout.py --config my_custom_config --checkpoint-path /path/to/your/model.pth --input-nc-path /path/to/your/initial_state.nc --output-nc-path /path/to/your/output.nc
```