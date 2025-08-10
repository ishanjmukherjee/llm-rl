"""
GRPO training script with configurable rewards.

Usage:
    python train.py --config configs/llama2-7b-gsm8k.yaml
"""

import argparse
import json
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from accelerate import Accelerator
import wandb
from tqdm import tqdm

# Import reward functions and callbacks
from rewards import get_reward_functions
from callbacks import CallbackRegistry


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(config):
    """Load training dataset based on configuration."""
    train_data = []
    train_file = config['dataset']['train_file']

    print(f"Loading dataset from {train_file}...")
    with open(train_file, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading data"):
            item = json.loads(line)
            train_data.append({
                "prompt": item["prompt"],
                "numerical_answer": item["numerical_answer"]
            })

    return Dataset.from_list(train_data)


def setup_model_and_tokenizer(config, accelerator):
    """Initialize model and tokenizer based on configuration."""
    model_config = config['model']
    model_name = model_config['name']

    print(f"Loading model: {model_name}")

    # Parse torch dtype
    dtype_map = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    torch_dtype = dtype_map.get(model_config.get('torch_dtype', 'float32'))

    # Load model
    model_kwargs = {
        'torch_dtype': torch_dtype,
        'device_map': {"": accelerator.device},
        'trust_remote_code': model_config.get('trust_remote_code', False),
    }

    # Add attention implementation if specified
    if 'attn_implementation' in model_config:
        model_kwargs['attn_implementation'] = model_config['attn_implementation']

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get('trust_remote_code', False)
    )
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_training_args(config):
    """Create GRPOConfig from configuration."""
    train_config = config['training']

    # Create output directory name
    model_name = config['model']['name'].replace('/', '-')
    output_dir = train_config.get('output_dir', f"./grpo_{model_name}")

    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(train_config.get('learning_rate', 5e-6)),
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'constant_with_warmup'),
        per_device_train_batch_size=train_config.get('per_device_train_batch_size', 32),
        num_train_epochs=train_config.get('num_train_epochs', 1),
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        warmup_steps=train_config.get('warmup_steps', 10),
        logging_steps=train_config.get('logging_steps', 1),
        save_steps=train_config.get('save_steps', 200),
        eval_strategy=train_config.get('eval_strategy', 'no'),
        bf16=train_config.get('bf16', True),
        report_to=train_config.get('report_to', 'wandb'),
        remove_unused_columns=train_config.get('remove_unused_columns', False),
        num_generations=train_config.get('num_generations', 4),
        temperature=float(train_config.get('temperature', 0.7)),
        max_completion_length=train_config.get('max_completion_length', 512),
        max_prompt_length=train_config.get('max_prompt_length', 512),
        disable_dropout=train_config.get('disable_dropout', True),
        loss_type=train_config.get('loss_type', 'dr_grpo'),
        log_completions=train_config.get('log_completions', True),
    )


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='GRPO Training Script')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize accelerator
    accelerator = Accelerator()

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config, accelerator)

    # Initialize wandb
    if accelerator.is_main_process and config.get('wandb', {}).get('enabled', True):
        wandb_config = config.get('wandb', {})
        project = wandb_config.get('project', 'grpo-training')
        wandb.init(project=project, config=config)

    # Load dataset
    dataset = load_dataset(config)
    print(f"Dataset loaded: {len(dataset)} examples")

    # Get reward functions based on configuration
    reward_names = config.get('rewards', [])
    if not reward_names:
        raise ValueError("No reward functions specified in config!")

    print(f"Using reward functions: {reward_names}")
    reward_funcs = get_reward_functions(reward_names)

    # Create training arguments
    training_args = create_training_args(config)

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
    )

    # Add callbacks based on configuration
    callbacks = CallbackRegistry.create_callbacks(config, config_path=args.config)
    for callback in callbacks:
        trainer.add_callback(callback)

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    if accelerator.is_main_process:
        model_name = config['model']['name'].replace('/', '-')
        final_path = f"grpo_{model_name}_final"
        trainer.save_model(final_path)
        print(f"Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
