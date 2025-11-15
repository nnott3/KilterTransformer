"""
Create config.json for your KilterGPT model.
Combine training args and model hyperparameters into one config dict.
"""

import json

config = {
    # --- Training Arguments ---
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "dataloader_num_workers": 0,
    "dataloader_pin_memory": False,
    "eval_steps": 2000,
    "eval_strategy": "steps",
    "gradient_accumulation_steps": 1,
    "hub_strategy": "every_save",
    "learning_rate": 1e-5,
    "logging_dir": "/content/drive/MyDrive/KilterTransformer/models/climb_gpt_new/20251111_085355_gpt_defaultloss_augmented/logs",
    "logging_steps": 100,
    "logging_strategy": "steps",
    "lr_scheduler_type": "linear",
    "metric_for_best_model": "eval_loss",
    "num_train_epochs": 15,
    "output_dir": "/content/drive/MyDrive/KilterTransformer/models/climb_gpt_new/20251111_085355_gpt_defaultloss_augmented",
    "overwrite_output_dir": True,
    "parallelism_config": None,
    "per_device_eval_batch_size": 8,
    "per_device_train_batch_size": 16,
    "push_to_hub": False,
    "push_to_hub_model_id": None,
    "push_to_hub_token": "<PUSH_TO_HUB_TOKEN>",
    "report_to": ["wandb"],
    "run_name": "20251111_085355_gpt_defaultloss_augmented",
    "save_steps": 2000,
    "save_strategy": "best",
    "save_total_limit": 2,
    "save_safetensors": False,
    "weight_decay": 0.01,
    "greater_is_better": False,
    "seed": 42,

    # --- Model & WandB Hyperparameters ---
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 6,
    "n_positions": 128,
    "dropout": 0.1,
    "epochs": 20,
    "batch_size": 16,
    "early_stopping_patience": 5,
    "allow_empty_prompt": True,
    "min_prefix_len": 1
}
checkpoint_dir = "/Users/nottreepat/Downloads/checkpoint-78000"
with open(f"{checkpoint_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2)

