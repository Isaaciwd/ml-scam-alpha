
import torch

# --- Machine/Environment ---
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# num_workers = 27
# prefetch_factor = 4
# persistent_workers = True

# --- Random Seed ---
seed = 42

# --- Model Architecture ---
model_architecture = "NNorm_LatentGlobalAtmosMixer" # Options: "LatentGlobalAtmosMixer", "NNorm_LatentGlobalAtmosMixer"
model_name = "NNormLatentGlobalAtmosMixer_small_MAE-AR2.pth"

# --- Data ---
# data_path_train = "/glade/derecho/scratch/idavis/archive/held_suarez/atm/hist/held_suarez.cam.h0i.*.nc"
# data_path_test = "/glade/derecho/scratch/idavis/archive/held_suarez/atm/hist/test/held_suarez.cam.h0i.*.nc"
# stats_path = "/glade/derecho/scratch/idavis/archive/held_suarez/atm/hist/stats/"
hours_later = 0.5
num_ar_steps = 2
variable_list = ["T", "U", "V", "Z3"]
tendency_variances_path = "/glade/derecho/scratch/idavis/archive/held_suarez/atm/hist/stats/variances_of_tendencies.pt"

# --- Training ---
training_mode = 'global'  # Options: 'global', 'single_column'
batch_size = 2
epochs = 100
lr = 2e-5
weight_decay = 0.02
l1_target_ratio = 0.03
grad_clip_max_norm = 1.0
use_mixed_precision = True
teacher_forcing_requires_grad = True
grad_ckpt_level = 1

# --- Single Column Training ---
# When training_mode is 'single_column', the train_model function from 
# train_offline_latent.py will be called instead of the global training function.
single_column_config = {
    "model_name": "NNormLatentGlobalAtmosMixer_small_MAE-AR2.pth",
    "td_stem": "latent/window_3",
    "batch_size": 256,
    "eval_steps": 5184 * 4,
    "patience": 150,
    "warmup_start_lr": 1e-5,
    "warmup_eval_cycles_scalar": 5,
    "lr": 2e-4,
    "epochs": 100,
    "num_workers": 5,
    "prefetch_factor": 4,
    "persistent_workers": True,
    "load": True,
    "resume_training": False,
    "resume_optimizer_use_new_lr_config": True,
    "reset_best_loss_on_load": True,
    "train_data_stem": "/glade/derecho/scratch/idavis/ml_scam/data/held_suarez/{td_stem}",
    "stats_path": "/glade/derecho/scratch/idavis/archive/held_suarez/atm/hist/stats/",
    "use_mixed_precision": False,
    "weight_decay": 0.01,
    "l1_target_ratio": 0.05,
    "cosine_loss_weight": 1.0,
    "use_dynamic_cosine_scaling": False,
    "print_interval_timing": True,
    "use_torch_compile": True,
    "grad_clip_max_norm": 1,
    "teacher_forcing_requires_grad": True,
    "grad_ckpt_level": 0,
    "loss_component_weights": {
        'representation': 1.0,
        'true_decoder': 1.0,
        'full_state': 1.0,
        'true_tendency': 1.0,
        'pred_tendency': 2.0,
    },
    "noise_factor": 0,
    "warmup_start_noise_factor": 0.0,
    "final_noise_factor": 0,
    "noise_std_dev_file": "/glade/work/idavis/ml_scam/TCNN/hybrid_cnn_transformer_v1_256_conv_correct_loss_no_cmpl_ckpt3_error_std_dev.nc",
    "noise_surface_var_names": ['PS', 'SOLIN', 'SST', 'TREFHT'],
    "hydrostatic_lambda": 0.0,
    "lapse_rate_lambda": 0.0,
    "ds_pl_var_list": ["T", "U", "V", "Z3"],
    "loss_type": 'MAE',
    "barlow_lambda": 5e-3,
    "mse_loss_type": 'mae',
    "scheduler_type": 'LinearDecayLR',
    "scheduler_patience": 5,
    "scheduler_factor": 0.6,
    "T_0": 5184 * 4 * 50,
    "T_mult": 2,
    "eta_min_cosine": 1e-8,
    "cosine_lr_t_max_batches": 5184 * 4 * 23,
    "linear_decay_total_batches": 5184 * 4 * 20,
    "linear_decay_end_lr": 1e-5,
}


# --- Model Loading ---
load_model = True
resume_training = False
resume_optimizer_use_new_lr_config = False
reset_best_loss_on_load = True
override_dropout = 0.0

# --- Loss Functions ---
loss_type = 'barlow'
mse_loss_type = 'mae'
barlow_lambda = 5e-2
loss_component_weights = {
    'representation': 1.5,
    'true_decoder': 1.0,
    'full_state': 1.0,
    'true_tendency': 4.0,
    'pred_tendency': 4.0,
    'ccc': 0.0,
    'ccc_latent': 0.0,
    'ccc_full_field': 4.0,
}

# --- Scheduler ---
scheduler_type = 'LinearDecayLR'
scheduler_patience = 5
scheduler_factor = 0.6
T_0 = 14 * 35
T_mult = 2
eta_min_cosine = 2e-8
cosine_lr_t_max_batches = 14 * 50
linear_decay_total_batches = 14 * 50
linear_decay_end_lr = 5e-6

# --- Warmup ---
warmup_eval_cycles_scalar = 1
warmup_start_lr = 1e-7

# --- Noise Injection ---
noise_factor = 0.0
warmup_start_noise_factor = 0.0
final_noise_factor = 0.0
# noise_std_dev_file = "/glade/work/idavis/ml_scam/TCNN/hybrid_cnn_transformer_v1_256_conv_correct_loss_no_cmpl_ckpt3_error_std_dev.nc"
noise_surface_var_names = ['PS', 'SOLIN', 'SST', 'TREFHT']

# --- Physics/Regularization ---
hydrostatic_lambda = 0.0
lapse_rate_lambda = 0.0

# --- Evaluation ---
eval_steps = 14

# --- Model Configuration ---
model_configs = {
    "LatentGlobalAtmosMixer": {
        'M_features': 122,
        'model_dim': 128,
        'n_heads': 32,
        'num_mixer_blocks': 4,
        'head_mlp_hidden_dims': [32],
        'ffn_hidden_dims': [1024],
        'encoder_hidden_dims': [256],
        'dropout_rate': 0.05,
    },
    "NNorm_LatentGlobalAtmosMixer": {
        'M_features': 122,
        'model_dim': 128,
        'n_heads': 32,
        'num_mixer_blocks': 4,
        'head_mlp_hidden_dims': [32],
        'ffn_hidden_dims': [1024],
        'encoder_hidden_dims': [256],
        'dropout_rate': 0.05,
    },

}
