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
import psutil
import os
from pathlib import Path
import importlib

from torch_funcs_global import train_model as train_global_model, evaluate_loss
from offline_torch_funcs_latent import train_model as train_offline_model
from GlobalAtmoMixr import LatentGlobalAtmosMixer
from NNorm_GAM import NNorm_LatentGlobalAtmosMixer
from global_data_loader import TemporalNetCDFDataset, ARTemporalNetCDFDataset

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
            return data_chunk
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise e

def print_memory_usage(label: str):
    process = psutil.Process(os.getpid())
    rss_memory_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[MEM_DEBUG] {label}: RSS Memory = {rss_memory_gb:.3f} GB")
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"[MEM_DEBUG] {label}: CUDA Memory Allocated = {allocated_gb:.3f} GB, Reserved = {reserved_gb:.3f} GB")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # --- Model Architecture Selection ---
    model_architecture = config.model_architecture
    model_name = config.model_name

    print(f"Model architecture: {model_architecture}")
    print(f"Model name: {model_name}")

    # Data paths
    stats_path = Path(getattr(config, 'stats_path', './stats'))
    
    # --- Model Configuration ---
    supported_architectures = {
        "LatentGlobalAtmosMixer": LatentGlobalAtmosMixer,
        "NNorm_LatentGlobalAtmosMixer": NNorm_LatentGlobalAtmosMixer,
    }
    
    model_class = supported_architectures.get(model_architecture)
    if not model_class:
        raise ValueError(f"Unsupported model_architecture '{model_architecture}'.")

    model_config = config.model_configs.get(model_architecture, {})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # --- Model Initialization ---
    model = model_class(**model_config).to(device)

    if config.training_mode == 'single_column':
        sc_config = config.single_column_config
        train_data_path = sc_config['train_data_stem'].format(td_stem=sc_config['td_stem'])
        
        files_train = f"{train_data_path}/data_dict_*.pt"
        files_test = f"{train_data_path}/test/data_dict_*pt"

        train_dataset = MultiComponentChunkedDataset(files_train)
        test_dataset = MultiComponentChunkedDataset(files_test)

        train_loader = DataLoader(
            train_dataset, 
            batch_size=1,
            shuffle=True, 
            num_workers=sc_config['num_workers'], 
            pin_memory=True,
            prefetch_factor=sc_config['prefetch_factor'],
            persistent_workers=sc_config['persistent_workers'],
            worker_init_fn=seed_worker
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=1, 
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=sc_config['persistent_workers'],
            worker_init_fn=seed_worker
        )
        
        train_offline_model(
            model, train_loader, test_loader, device, model_config,
            epochs=sc_config['epochs'], lr=sc_config['lr'], batch_size=sc_config['batch_size'],
            eval_steps=sc_config['eval_steps'], patience=sc_config['patience'],
            model_path=f'./models/{sc_config["model_name"]}',
            l1_target_ratio=sc_config['l1_target_ratio'],
            warmup_eval_steps=sc_config['warmup_eval_cycles_scalar'] * sc_config['eval_steps'],
            warmup_start_lr=sc_config['warmup_start_lr'],
            scheduler_type=sc_config['scheduler_type'],
            scheduler_patience=sc_config['scheduler_patience'],
            scheduler_factor=sc_config['scheduler_factor'], T_0=sc_config['T_0'],
            T_mult=sc_config['T_mult'], eta_min_cosine=sc_config['eta_min_cosine'],
            cosine_lr_t_max_batches=sc_config['cosine_lr_t_max_batches'],
            linear_decay_total_batches=sc_config['linear_decay_total_batches'],
            linear_decay_end_lr=sc_config['linear_decay_end_lr'],
            use_mixed_precision=sc_config['use_mixed_precision'],
            loaded_checkpoint=None, resume_training=sc_config['resume_training'],
            resume_optimizer_use_new_lr_config=sc_config['resume_optimizer_use_new_lr_config'],
            reset_best_loss_on_load=sc_config['reset_best_loss_on_load'],
            weight_decay=sc_config['weight_decay'],
            hydrostatic_lambda=sc_config['hydrostatic_lambda'],
            lapse_rate_lambda=sc_config['lapse_rate_lambda'],
            ds_pl_var_list=sc_config['ds_pl_var_list'],
            print_interval_timing=sc_config['print_interval_timing'],
            use_torch_compile=sc_config['use_torch_compile'],
            grad_clip_max_norm=sc_config['grad_clip_max_norm'],
            noise_factor=sc_config['noise_factor'],
            warmup_start_noise_factor=sc_config['warmup_start_noise_factor'],
            final_noise_factor=sc_config['final_noise_factor'],
            noise_std_dev_file=sc_config.get('noise_std_dev_file'),
            noise_surface_var_names=sc_config['noise_surface_var_names'],
            cosine_loss_weight=sc_config['cosine_loss_weight'],
            use_dynamic_cosine_scaling=sc_config['use_dynamic_cosine_scaling'],
            loss_type=sc_config['loss_type'],
            barlow_lambda=sc_config['barlow_lambda'],
            loss_weights=sc_config['loss_component_weights'],
            mse_loss_type=sc_config['mse_loss_type'],
            tendency_variances_path=config.tendency_variances_path,
        )

    elif config.training_mode == 'global':
        ds_path_train = getattr(config, 'data_path_train', None)
        ds_path_test = getattr(config, 'data_path_test', None)
        
        if ds_path_train and ds_path_test:
            train_dataset = ARTemporalNetCDFDataset(
                ds_path_pattern=ds_path_train,
                hours_later=config.hours_later,
                num_ar_steps=config.num_ar_steps,
                variable_list=config.variable_list,
                dtype=torch.float32,
                stats_path=stats_path
            )
            test_dataset = ARTemporalNetCDFDataset(
                ds_path_pattern=ds_path_test,
                hours_later=config.hours_later,
                num_ar_steps=config.num_ar_steps,
                variable_list=config.variable_list,
                dtype=torch.float32,
                stats_path=stats_path
            )
        else:
            raise NotImplementedError("Global training data loading not yet implemented in this script.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=getattr(config, 'num_workers', 4),
            pin_memory=True,
            prefetch_factor=getattr(config, 'prefetch_factor', 2),
            persistent_workers=getattr(config, 'persistent_workers', False),
            worker_init_fn=seed_worker
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=getattr(config, 'num_workers', 4),
            pin_memory=True,
            prefetch_factor=getattr(config, 'prefetch_factor', 2),
            persistent_workers=getattr(config, 'persistent_workers', False),
            worker_init_fn=seed_worker
        )
        
        train_global_model(
            model, train_loader, test_loader, device, model_config,
            epochs=config.epochs, lr=config.lr, batch_size=config.batch_size, eval_steps=config.eval_steps,
            patience=config.patience, model_path=f'./models/{model_name}',
            l1_target_ratio=config.l1_target_ratio,
            warmup_eval_steps=config.warmup_eval_cycles_scalar * config.eval_steps,
            warmup_start_lr=config.warmup_start_lr,
            scheduler_type=config.scheduler_type, scheduler_patience=config.scheduler_patience,
            scheduler_factor=config.scheduler_factor, T_0=config.T_0, T_mult=config.T_mult,
            eta_min_cosine=config.eta_min_cosine,
            cosine_lr_t_max_batches=config.cosine_lr_t_max_batches,
            linear_decay_total_batches=config.linear_decay_total_batches,
            linear_decay_end_lr=config.linear_decay_end_lr,
            use_mixed_precision=config.use_mixed_precision,
            loaded_checkpoint=None, # Simplified for now
            resume_training=config.resume_training,
            resume_optimizer_use_new_lr_config=config.resume_optimizer_use_new_lr_config,
            reset_best_loss_on_load=config.reset_best_loss_on_load,
            weight_decay=config.weight_decay,
            hydrostatic_lambda=config.hydrostatic_lambda,
            lapse_rate_lambda=config.lapse_rate_lambda,
            ds_pl_var_list=config.variable_list,
            print_interval_timing=True,
            use_torch_compile=False,
            grad_clip_max_norm=config.grad_clip_max_norm,
            noise_factor=config.noise_factor,
            warmup_start_noise_factor=config.warmup_start_noise_factor,
            final_noise_factor=config.final_noise_factor,
            noise_std_dev_file=getattr(config, 'noise_std_dev_file', None),
            noise_surface_var_names=config.noise_surface_var_names,
            cosine_loss_weight=1.0, # Simplified
            use_dynamic_cosine_scaling=False, # Simplified
            loss_type=config.loss_type,
            barlow_lambda=config.barlow_lambda,
            loss_weights=config.loss_component_weights,
            use_accelerate=False, # Simplified
            accelerate_kwargs={}, # Simplified
            mse_loss_type=config.mse_loss_type,
            tendency_variances_path=config.tendency_variances_path,
        )
    else:
        raise ValueError(f"Invalid training_mode: {config.training_mode}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train latent mixer model")
    parser.add_argument("--config", type=str, default="config", help="Configuration file to use")
    args = parser.parse_args()

    # Import the configuration file
    try:
        config_module = importlib.import_module(args.config)
    except ImportError:
        print(f"Error: Could not import configuration file {args.config}.py")
        sys.exit(1)

    main(config_module)