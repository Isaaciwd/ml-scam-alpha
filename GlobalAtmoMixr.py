#%%
import logging
logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
import os
os.environ['TORCH_LOGS'] = '-fake_tensor'
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional
import numpy as np

# ==============================================================================
# === Core Mixer Components (Combined from both models)                      ===
# ==============================================================================

class HeadMLP(nn.Module):
    """
    An MLP for a single 'head' in a STANDARD mixer block.
    It mixes spatial information for each channel and returns the updated spatial tokens.
    """
    def __init__(self, num_patches: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        current_dim = num_patches
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, num_patches))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size * head_dim, num_patches)
        # Output shape: (batch_size * head_dim, num_patches)
        return self.mlp(x)

class CentralColumnHeadMLP(nn.Module):
    """
    An MLP for a single 'head' in the FINAL mixer block.
    It takes spatial info and computes a single scalar for the central column.
    """
    def __init__(self, num_patches: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        current_dim = num_patches
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, 1)) # Output is a single value
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch_size * head_dim, num_patches)
        # Output shape: (batch_size * head_dim, 1)
        return self.mlp(x)

class StandardMixerBlock(nn.Module):
    """
    A standard mixer block that processes and updates ALL tokens (grid columns) in a patch.
    This is used for intermediate layers.
    """
    def __init__(self, model_dim: int, patch_size: int, n_heads: int,
                 head_mlp_hidden_dims: List[int], ffn_hidden_dims: List[int], dropout_rate: float,
                 use_residual: bool = True):
        super().__init__()
        self.head_dim = model_dim // n_heads
        self.n_heads = n_heads
        num_patches = patch_size * patch_size
        self.use_residual = use_residual
        self.grad_ckpt_level = 0

        self.heads = nn.ModuleList([
            HeadMLP(num_patches=num_patches, hidden_dims=head_mlp_hidden_dims)
            for _ in range(n_heads)
        ])

        ffn_layers = []
        current_dim = model_dim
        for h_dim in ffn_hidden_dims:
            ffn_layers.append(nn.Linear(current_dim, h_dim))
            ffn_layers.append(nn.GELU())
            current_dim = h_dim
        ffn_layers.append(nn.Linear(current_dim, model_dim))
        ffn_layers.append(nn.Dropout(dropout_rate))
        self.ffn = nn.Sequential(*ffn_layers)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level

    def forward(self, x):
        # Input x shape: (batch_of_patches, num_patches, model_dim)
        x_res = x
        x = self.norm1(x)

        # Split into heads for spatial mixing
        x_split = torch.split(x, self.head_dim, dim=-1)
        head_outputs = []
        for i in range(self.n_heads):
            head_chunk = x_split[i].permute(0, 2, 1) # -> (B, head_dim, num_patches)
            B_p, C_h, N_p = head_chunk.shape
            head_chunk_flat = head_chunk.reshape(-1, N_p)

            if (
                self.grad_ckpt_level >= 1
                and self.training
                and head_chunk_flat.requires_grad
            ):
                processed_flat = checkpoint(self.heads[i], head_chunk_flat, use_reentrant=False)
            else:
                processed_flat = self.heads[i](head_chunk_flat) # -> (B*C, N_p)

            processed_chunk = processed_flat.reshape(B_p, C_h, N_p).permute(0, 2, 1) # -> (B, N_p, C)
            head_outputs.append(processed_chunk)

        x = torch.cat(head_outputs, dim=-1)
        if self.use_residual:
            x = x + x_res # First residual connection

        x_res = x
        x = self.norm2(x)
        x = self.ffn(x) # Channel mixing
        if self.use_residual:
            x = x + x_res # Second residual connection
        return x

class FinalMixerBlock(nn.Module):
    """
    A specialized mixer block that only computes the feature for the central column.
    This is used as the final layer in the mixer sequence.
    """
    def __init__(self, model_dim: int, patch_size: int, n_heads: int,
                 head_mlp_hidden_dims: List[int], ffn_hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        self.head_dim = model_dim // n_heads
        self.n_heads = n_heads
        num_patches = patch_size * patch_size
        self.grad_ckpt_level = 0

        self.center_only_heads = nn.ModuleList([
            CentralColumnHeadMLP(num_patches=num_patches, hidden_dims=head_mlp_hidden_dims)
            for _ in range(n_heads)
        ])

        ffn_layers = []
        current_dim = model_dim
        for h_dim in ffn_hidden_dims:
            ffn_layers.append(nn.Linear(current_dim, h_dim))
            ffn_layers.append(nn.GELU())
            current_dim = h_dim
        ffn_layers.append(nn.Linear(current_dim, model_dim))
        ffn_layers.append(nn.Dropout(dropout_rate))
        self.ffn = nn.Sequential(*ffn_layers)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def set_grad_ckpt_level(self, level: int) -> None:
        self.grad_ckpt_level = level

    def forward(self, x):
        # Input x shape: (batch_of_patches, num_patches, model_dim)
        central_idx = (x.shape[1] - 1) // 2
        x_res = x[:, central_idx, :]

        x_norm = self.norm1(x)
        x_split = torch.split(x_norm, self.head_dim, dim=-1)

        head_outputs = []
        for i in range(self.n_heads):
            head_chunk = x_split[i].permute(0, 2, 1) # -> (B, head_dim, num_patches)
            B_p, C_h, N_p = head_chunk.shape
            head_chunk_flat = head_chunk.reshape(-1, N_p)
            
            if (
                self.grad_ckpt_level >= 1
                and self.training
                and head_chunk_flat.requires_grad
            ):
                center_val_flat = checkpoint(self.center_only_heads[i], head_chunk_flat, use_reentrant=False)
            else:
                center_val_flat = self.center_only_heads[i](head_chunk_flat) # -> (B*C, 1)

            center_val = center_val_flat.reshape(B_p, C_h)
            head_outputs.append(center_val)

        x_mixed = torch.cat(head_outputs, dim=-1)
        x = x_mixed + x_res # First residual connection

        x_res = x
        x = self.norm2(x)
        x = self.ffn(x) # Channel mixing
        return x + x_res # Second residual connection

# ==============================================================================
# === Encoder / Decoder Components                                         ===
# ==============================================================================

class Encoder(nn.Module):
    """Encodes the physical state into a latent space at each grid point."""
    def __init__(self, M_features, model_dim, encoder_hidden_dims: List[int]):
        super().__init__()
        layers = []
        current_dim = M_features
        for h_dim in encoder_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, model_dim))
        layers.append(nn.LayerNorm(model_dim))
        self.encoder_mlp = nn.Sequential(*layers)

    def forward(self, x):
        # Input can be (B, H, W, M) or (B*H*W, M)
        original_shape = x.shape
        if x.dim() == 4:
            B, H, W, M = x.shape
            x_flat = x.view(-1, M)
            encoded_flat = self.encoder_mlp(x_flat)
            return encoded_flat.view(B, H, W, -1)
        elif x.dim() == 2:
            return self.encoder_mlp(x)
        else:
            raise ValueError(f"Unsupported input dimension for Encoder: {x.dim()}")

class Decoder(nn.Module):
    """Decodes the latent state back to the physical state at each grid point."""
    def __init__(self, model_dim, M_features, decoder_hidden_dims: List[int]):
        super().__init__()
        layers = []
        current_dim = model_dim
        for h_dim in decoder_hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.GELU())
            current_dim = h_dim
        layers.append(nn.Linear(current_dim, M_features))
        self.decoder_mlp = nn.Sequential(*layers)

    def forward(self, z):
        # Input can be (B, H, W, D) or (B*H*W, D)
        if z.dim() > 2:
             B, H, W, D = z.shape
             z_flat = z.view(-1, D)
             decoded_flat = self.decoder_mlp(z_flat)
             return decoded_flat.view(B, H, W, -1)
        elif z.dim() == 2:
             return self.decoder_mlp(z)
        else:
             raise ValueError(f"Unsupported input dimension for Decoder: {z.dim()}")


# ==============================================================================
# === Main Global Latent Model (Refactored)                                  ===
# ==============================================================================

class LatentGlobalAtmosMixer(nn.Module):
    """
    A global atmospheric model that learns tendencies in a latent space.

    It encodes a global grid into a latent representation, then applies a
    sequence of mixer blocks to predict the latent tendency for every grid cell.
    The final latent state is decoded back to the physical state.

    """
    def __init__(self, M_features: int, model_dim: int, n_heads: int,
                 num_mixer_blocks: int, head_mlp_hidden_dims: List[int],
                 ffn_hidden_dims: List[int], encoder_hidden_dims: List[int],
                 dropout_rate: float = 0.1, grad_ckpt_level: int = 1, **kwargs):
        super().__init__()
        self.M_features = M_features
        self.model_dim = model_dim
        self.default_num_rollout_steps = 1
        self.default_decode_freq = 1
        self.is_training_mode = True
        self.teacher_forcing_requires_grad = kwargs.pop('teacher_forcing_requires_grad', True)
        self.grad_ckpt_level = grad_ckpt_level
        if kwargs:
            unexpected = ", ".join(kwargs.keys())
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if self.grad_ckpt_level not in (0, 1, 2):
            raise ValueError("grad_ckpt_level must be one of {0, 1, 2}.")

        if num_mixer_blocks < 1:
            raise ValueError("num_mixer_blocks must be at least 1.")

        # --- 1. Encoder and Decoder ---
        self.encoder = Encoder(M_features, model_dim, encoder_hidden_dims)
        self.decoder = Decoder(model_dim, M_features, encoder_hidden_dims[::-1]) # Symmetrical decoder
        self.tend_decoder = Decoder(model_dim, M_features, encoder_hidden_dims[::-1]) # Symmetrical decoder


        # --- 2. Mixer Blocks ---
        mixer_blocks_list = []
        # Add N-1 Standard Mixer Blocks
        for block_idx in range(num_mixer_blocks - 1):
            mixer_blocks_list.append(
                StandardMixerBlock(
                    model_dim,
                    3,
                    n_heads,
                    head_mlp_hidden_dims,
                    ffn_hidden_dims,
                    dropout_rate,
                    use_residual=(block_idx != 0)
                )
            )
        # Add the Final Mixer Block to collapse the spatial info
        mixer_blocks_list.append(
            FinalMixerBlock(model_dim, 3, n_heads, head_mlp_hidden_dims, ffn_hidden_dims, dropout_rate)
        )
        self.mixer = nn.Sequential(*mixer_blocks_list)
        self._set_mixer_grad_ckpt_level(self.grad_ckpt_level)

    def set_teacher_forcing_grad(self, enabled: bool) -> None:
        self.teacher_forcing_requires_grad = bool(enabled)

    def set_grad_ckpt_level(self, level: int) -> None:
        if level not in (0, 1, 2):
            raise ValueError("grad_ckpt_level must be one of {0, 1, 2}.")
        self.grad_ckpt_level = level
        self._set_mixer_grad_ckpt_level(level)

    def _set_mixer_grad_ckpt_level(self, level: int) -> None:
        for block in self.mixer:
            if hasattr(block, "set_grad_ckpt_level"):
                block.set_grad_ckpt_level(level)

    def forward(
        self,
        x_current_global: torch.Tensor,
        x_next_global: Optional[torch.Tensor] = None,
        *,
        num_rollout_steps: Optional[int] = None,
        decode_freq: Optional[int] = None,
        train: Optional[bool] = None,
        global_mode: Optional[bool] = True
    ):
        use_full_ckpt = (
            self.grad_ckpt_level >= 2
            and global_mode
            and self.training
        )

        if use_full_ckpt:
            return self._forward_with_checkpoint(
                x_current_global,
                x_next_global,
                num_rollout_steps=num_rollout_steps,
                decode_freq=decode_freq,
                train=train,
                global_mode=global_mode,
            )

        return self._forward_impl(
            x_current_global,
            x_next_global,
            num_rollout_steps=num_rollout_steps,
            decode_freq=decode_freq,
            train=train,
            global_mode=global_mode,
        )

    def _forward_with_checkpoint(
        self,
        x_current_global: torch.Tensor,
        x_next_global: Optional[torch.Tensor] = None,
        *,
        num_rollout_steps: Optional[int] = None,
        decode_freq: Optional[int] = None,
        train: Optional[bool] = None,
        global_mode: Optional[bool] = True,
    ):
        use_next_placeholder = x_next_global is None
        if use_next_placeholder:
            x_next_arg = torch.zeros(1, device=x_current_global.device, dtype=x_current_global.dtype)
        else:
            x_next_arg = x_next_global

        dummy = torch.zeros(1, device=x_current_global.device, requires_grad=True)

        def body(x_cur: torch.Tensor, x_next: torch.Tensor, _dummy: torch.Tensor):
            next_tensor = None if use_next_placeholder else x_next
            outputs = self._forward_impl(
                x_cur,
                next_tensor,
                num_rollout_steps=num_rollout_steps,
                decode_freq=decode_freq,
                train=train,
                global_mode=global_mode,
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            packed, masks = self._pack_checkpoint_outputs(outputs, reference_tensor=x_cur)
            return tuple(packed + masks)

        checkpoint_outputs = checkpoint(body, x_current_global, x_next_arg, dummy, use_reentrant=False)
        restored = self._unpack_checkpoint_outputs(checkpoint_outputs)
        return restored if isinstance(restored, tuple) else tuple(restored)

    def _pack_checkpoint_outputs(self, outputs: tuple[torch.Tensor, ...], reference_tensor: torch.Tensor):
        packed = []
        masks = []
        for out in outputs:
            if out is None:
                placeholder = torch.zeros(1, device=reference_tensor.device, dtype=reference_tensor.dtype)
                packed.append(placeholder)
                masks.append(torch.zeros(1, device=reference_tensor.device))
            else:
                packed.append(out)
                mask = torch.ones(1, device=out.device)
                masks.append(mask)
        return packed, masks

    def _unpack_checkpoint_outputs(self, checkpoint_outputs: tuple[torch.Tensor, ...]):
        half = len(checkpoint_outputs) // 2
        values = checkpoint_outputs[:half]
        masks = checkpoint_outputs[half:]
        restored = []
        for tensor, mask in zip(values, masks):
            if mask.item() == 0:
                restored.append(None)
            else:
                restored.append(tensor)
        return tuple(restored)


    def _apply_custom_padding(self, x):
        # Input x shape: (B, H, W, C)
        # Use circular padding for longitude (dim 2 of CHW format)
        x_circ = F.pad(x.permute(0, 3, 1, 2), (1, 1, 0, 0), mode='circular').permute(0, 2, 3, 1)
        
        # Flip poles for latitude padding
        north_pole_row = torch.flip(x_circ[:, 0:1, :, :], dims=[2])
        south_pole_row = torch.flip(x_circ[:, -1:, :, :], dims=[2])
        
        x_padded = torch.cat([north_pole_row, x_circ, south_pole_row], dim=1)
        return x_padded

    def _compute_latent_delta(self, latent_grid: torch.Tensor) -> torch.Tensor:
        if latent_grid.dim() != 4:
            raise ValueError("latent_grid must have shape (batch, lat, lon, model_dim).")

        batch, height, width, _ = latent_grid.shape

        padded = self._apply_custom_padding(latent_grid)
        padded_permuted = padded.permute(0, 3, 1, 2)

        patches = padded_permuted.unfold(2, 3, 1).unfold(3, 3, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.model_dim, 9)
        patches = patches.permute(0, 2, 1)

        # Apply gradient checkpointing to each mixer block individually
        delta_flat = patches
        use_block_ckpt = self.grad_ckpt_level >= 1
        for mixer_block in self.mixer:
            if use_block_ckpt and self.training and delta_flat.requires_grad:
                delta_flat = checkpoint(mixer_block, delta_flat, use_reentrant=False)
            else:
                delta_flat = mixer_block(delta_flat)
        
        return delta_flat.view(batch, height, width, self.model_dim)

    def _forward_impl(
        self,
        x_current_global: torch.Tensor,
        x_next_global: Optional[torch.Tensor] = None,
        *,
        num_rollout_steps: Optional[int] = None,
        decode_freq: Optional[int] = None,
        train: Optional[bool] = None,
        global_mode: Optional[bool] = True
    ):
        # if in global mode, do forward pass wiht infrasturcture for global prediction and rollout
        if global_mode:
            if x_current_global.dim() != 4:
                raise ValueError("x_current_global must have shape (batch, lat, lon, features).")

            batch, height, width, features = x_current_global.shape
            if features != self.M_features:
                raise ValueError(
                    f"x_current_global last dimension ({features}) must match initialized M_features ({self.M_features})."
                )

            if num_rollout_steps is None:
                steps = self.default_num_rollout_steps
            else:
                if num_rollout_steps < 1:
                    raise ValueError("num_rollout_steps must be at least 1.")
                steps = num_rollout_steps
                self.default_num_rollout_steps = num_rollout_steps

            if decode_freq is None:
                decode_frequency = self.default_decode_freq
            else:
                if decode_freq < 1:
                    raise ValueError("decode_freq must be at least 1.")
                decode_frequency = decode_freq
                self.default_decode_freq = decode_freq

            if train is None:
                training_mode = self.is_training_mode
            else:
                training_mode = bool(train)
                self.is_training_mode = training_mode

            if training_mode:
                if x_next_global is None:
                    raise ValueError("x_next_global is required when the model is in training mode.")
                if x_next_global.dim() == 4 and steps == 1:
                    x_next_global = x_next_global.unsqueeze(1)
                elif x_next_global.dim() != 5:
                    raise ValueError(
                        "x_next_global must have shape (batch, num_rollout_steps, lat, lon, features) in training mode."
                    )
                if x_next_global.shape[0] != batch:
                    raise ValueError("x_next_global batch dimension must match x_current_global.")
                if x_next_global.shape[1] != steps:
                    raise ValueError("x_next_global num_rollout_steps dimension must match the provided value.")
                if x_next_global.shape[2] != height or x_next_global.shape[3] != width:
                    raise ValueError("Spatial dimensions of x_next_global must match x_current_global.")
                if x_next_global.shape[4] != self.M_features:
                    raise ValueError("x_next_global feature dimension must match initialized M_features.")

            encoded_current = self.encoder(x_current_global)
            latent_state = encoded_current

            x_next_pred_steps: List[torch.Tensor] = []
            delta_pred_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
            delta_true_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
            reconstructed_true_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
            tend_pred_steps: Optional[List[torch.Tensor]] = [] if training_mode else None
            tend_true_steps: Optional[List[torch.Tensor]] = [] if training_mode else None

            if training_mode and x_next_global is not None:
                x_next_flat = x_next_global.reshape(batch * steps, height, width, self.M_features)
                with torch.set_grad_enabled(self.teacher_forcing_requires_grad):
                    encoded_next_true_flat = self.encoder(x_next_flat)
                    encoded_next_true = encoded_next_true_flat.reshape(batch, steps, height, width, self.model_dim)

                    reconstructed_true_flat = self.decoder(encoded_next_true_flat)
                    reconstructed_true = reconstructed_true_flat.reshape(batch, steps, height, width, self.M_features)

                if not self.teacher_forcing_requires_grad:
                    encoded_next_true = encoded_next_true.detach()
                    reconstructed_true = reconstructed_true.detach()
            else:
                encoded_next_true = None
                reconstructed_true = None

            for step_idx in range(steps):
                delta_pred = self._compute_latent_delta(latent_state)
                pred_tend = self.tend_decoder(delta_pred)
                encoded_next_pred = latent_state + delta_pred

                should_decode = (step_idx + 1) % decode_frequency == 0
                if should_decode or (step_idx == steps - 1 and not x_next_pred_steps):
                    decoded_pred = self.decoder(encoded_next_pred)
                    x_next_pred_steps.append(decoded_pred)

                if training_mode and delta_pred_steps is not None:
                    delta_pred_steps.append(delta_pred)
                    tend_pred_steps.append(pred_tend)

                if (
                    training_mode
                    and delta_true_steps is not None
                    and reconstructed_true_steps is not None
                    and encoded_next_true is not None
                    and reconstructed_true is not None
                ):
                    encoded_next_true_step = encoded_next_true[:, step_idx, ...].contiguous()
                    if not self.teacher_forcing_requires_grad:
                        encoded_next_true_step = encoded_next_true_step.detach()

                    if step_idx == 0:
                        prev_true_latent = encoded_current
                    else:
                        prev_true_latent = encoded_next_true[:, step_idx - 1, ...].contiguous()
                    if not self.teacher_forcing_requires_grad:
                        prev_true_latent = prev_true_latent.detach()

                    # delta_true = encoded_next_true_step - prev_true_latent
                    delta_true = encoded_next_true_step - latent_state #DC
                    if not self.teacher_forcing_requires_grad:
                        delta_true = delta_true.detach()

                    if self.teacher_forcing_requires_grad:
                        true_tend = self.tend_decoder(delta_true)
                    else:
                        with torch.no_grad():
                            true_tend = self.tend_decoder(delta_true)
                    tend_true_steps.append(true_tend)
                    delta_true_steps.append(delta_true)
                    reconstructed_true_steps.append(reconstructed_true[:, step_idx, ...].contiguous())

                latent_state = encoded_next_pred

            if not x_next_pred_steps:
                decoded_pred = self.decoder(latent_state)
                x_next_pred_steps.append(decoded_pred)

            x_next_pred_rollout = torch.stack(x_next_pred_steps, dim=1)

            if training_mode:
                delta_pred_rollout = torch.stack(delta_pred_steps, dim=1) if delta_pred_steps else None
                if delta_true_steps:
                    delta_true_rollout = torch.stack(delta_true_steps, dim=1)
                else:
                    delta_true_rollout = None

                if reconstructed_true_steps:
                    reconstructed_true_rollout = torch.stack(reconstructed_true_steps, dim=1)
                else:
                    reconstructed_true_rollout = None
                if tend_pred_steps:
                    tend_pred_rollout = torch.stack(tend_pred_steps, dim=1)
                    tend_true_rollout = torch.stack(tend_true_steps, dim=1)
                else:
                    tend_pred_rollout = None
                    tend_true_rollout = None

                return (
                    delta_pred_rollout,
                    delta_true_rollout,
                    x_next_pred_rollout,
                    reconstructed_true_rollout,
                    tend_pred_rollout,
                    tend_true_rollout

                )

            return x_next_pred_rollout

        # if not in globa mode, use single column training flow.
        # x_current_global and x_next_global will be given as (B, 3, 3, M)
        else:
            # main model computation
            encoded_state = self.encoder(x_current_global)
            B, H, W, C = encoded_state.shape
            E_current_flat_patch = encoded_state.view(B, H * W, C)
            delta_pred = self.mixer(E_current_flat_patch)
            E_next = encoded_state[:,1,1] + delta_pred
            decoded_pred = self.decoder(E_next)

            # calculate auxilary outputs for loss computation
            encoded_next_true = self.encoder(x_next_global) # 
            delta_true = encoded_next_true - encoded_state[:,1,1]
            reconstructed_true = self.decoder(encoded_next_true)
            tend_pred = self.tend_decoder(delta_pred)
            tend_true = self.tend_decoder(delta_true)

            return delta_pred, delta_true, decoded_pred, reconstructed_true, tend_pred, tend_true



# ==============================================================================
# === Self-Contained Test Block                                              ===
# ==============================================================================

if __name__ == '__main__':
    # --- User-Selectable Hyperparameters ---
    BATCH_SIZE = 2
    N_LAT = 32   # Using smaller grid for faster testing
    N_LON = 64
    M_FEATURES = 122

    MODEL_DIM = 512
    N_HEADS = 32

    NUM_MIXER_BLOCKS = 6 # Must be >= 1

    # --- MLP Shape Configurations ---
    HEAD_MLP_HIDDEN_DIMS = [32]
    FFN_HIDDEN_DIMS = [MODEL_DIM * 2]
    ENCODER_HIDDEN_DIMS = []

    # --- Model Instantiation ---
    print("--- Latent Global Atmos Mixer Configuration ---")
    print(f"Grid size: {N_LAT}x{N_LON}")
    print(f"Input/Output features (M): {M_FEATURES}")
    print(f"Internal model dimension: {MODEL_DIM}")
    print(f"Encoder hidden layers: {ENCODER_HIDDEN_DIMS}")
    print(f"Number of mixer blocks (N-1 standard + 1 final): {NUM_MIXER_BLOCKS}")
    print("-" * 45)

    model = LatentGlobalAtmosMixer(
        M_features=M_FEATURES,
        model_dim=MODEL_DIM,
        n_heads=N_HEADS,
        num_mixer_blocks=NUM_MIXER_BLOCKS,
        head_mlp_hidden_dims=HEAD_MLP_HIDDEN_DIMS,
        ffn_hidden_dims=FFN_HIDDEN_DIMS,
        encoder_hidden_dims=ENCODER_HIDDEN_DIMS,
        dropout_rate=0.1
    )

    # --- Create Dummy Data ---
    dummy_x_current = torch.randn(BATCH_SIZE, N_LAT, N_LON, M_FEATURES)
    dummy_x_next = torch.randn(BATCH_SIZE, N_LAT, N_LON, M_FEATURES)

    print(f"\nInput tensor shapes (current/next): {dummy_x_current.shape}")

    # --- Perform a Forward Pass ---
    delta_E_pred, delta_E_true, x_next_pred, y = model(dummy_x_current, dummy_x_next)

    print("\n--- Output Shapes ---")
    print(f"Predicted Latent Tendency (delta_E_pred): {delta_E_pred.shape}")
    print(f"  > Expected: {(BATCH_SIZE, N_LAT, N_LON, MODEL_DIM)}")
    print(f"True Latent Tendency (delta_E_true):      {delta_E_true.shape}")
    print(f"  > Expected: {(BATCH_SIZE, N_LAT, N_LON, MODEL_DIM)}")
    print(f"Decoded Physical Prediction (x_next_pred): {x_next_pred.shape}")
    print(f"  > Expected: {(BATCH_SIZE, N_LAT, N_LON, M_FEATURES)}")


    # --- Calculate and Print Total Parameters ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

# %%

