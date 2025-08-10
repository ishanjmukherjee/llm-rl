#!/usr/bin/env python
"""
Test callback functionality without running full training.

Usage:
    python test_callbacks.py --config configs/llama2-7b-gsm8k.yaml
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
from transformers import TrainerState, TrainerControl
from transformers.training_args import TrainingArguments

from callbacks import CompletionLoggerCallback, AsyncMetricsPlotterCallback, CallbackRegistry


def test_callbacks(config_path):
    """Test callbacks with mock data."""

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Testing callbacks with mock data...\n")

    # Generate test run ID
    config_name = Path(config_path).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    test_run_id = f"{config_name}_test_{timestamp}"

    # Create mock training args and state
    mock_args = TrainingArguments(
        output_dir="test_output",
        logging_steps=1,
        save_steps=100,
    )

    mock_state = TrainerState()
    mock_state.global_step = 100

    mock_control = TrainerControl()

    # Test Completion Logger
    print("="*60)
    print("Testing CompletionLoggerCallback")
    print("="*60)

    completion_config = config.get('callbacks', {}).get('completion_logger', {})
    if completion_config.get('enabled', False):
        output_dir = completion_config.get('output_dir', 'completions_test')
        callback = CompletionLoggerCallback(
            output_dir=output_dir,
            run_id=test_run_id
        )

        # Add mock completions to state
        mock_state.log_history = [{
            'completions': [
                {'prompt': 'What is 2+2?', 'completion': 'The answer is 4.', 'reward': 1.0},
                {'prompt': 'What is 5*3?', 'completion': 'Let me calculate: 5*3 = 15', 'reward': 0.8},
            ]
        }]

        # Trigger save
        callback.on_save(mock_args, mock_state, mock_control)

        # Check results
        run_dir = Path(output_dir) / test_run_id
        if run_dir.exists():
            print(f"✓ Created run directory: {run_dir}")

            step_dir = run_dir / "step_100"
            if step_dir.exists():
                print(f"✓ Created step directory: {step_dir}")

                completion_file = step_dir / "completions.txt"
                if completion_file.exists():
                    print(f"✓ Created completion file: {completion_file}")

                latest_link = run_dir / "latest"
                if latest_link.exists() and latest_link.is_symlink():
                    print(f"✓ Created latest symlink -> {latest_link.resolve().name}")
        else:
            print("✗ Failed to create run directory")
    else:
        print("CompletionLoggerCallback is disabled in config")

    # Test Metrics Plotter
    print("\n" + "="*60)
    print("Testing AsyncMetricsPlotterCallback")
    print("="*60)

    plotter_config = config.get('callbacks', {}).get('metrics_plotter', {})
    if plotter_config.get('enabled', False):
        plot_dir = plotter_config.get('plot_dir', 'plots_test')
        callback = AsyncMetricsPlotterCallback(
            plot_dir=plot_dir,
            metrics_to_plot=plotter_config.get('metrics_to_plot', None),
            run_id=test_run_id
        )

        # Simulate multiple log events
        for step in range(10, 110, 10):
            mock_state.global_step = step
            mock_logs = {
                'train/grad_norm': 0.5 + step * 0.01,
                'train/reward_mean': 0.1 + step * 0.005,
                'train/reward_std': 0.2 - step * 0.001,
                'train/kl_divergence': 0.01 + step * 0.0001,
                'learning_rate': 5e-6 * (1 - step/200),
            }
            callback.on_log(mock_args, mock_state, mock_control, logs=mock_logs)

        # Trigger plot on save
        import time
        callback.on_save(mock_args, mock_state, mock_control)

        # Wait a moment for async plotting
        print("Waiting for async plotting to complete...")
        time.sleep(2)

        # Check results
        run_dir = Path(plot_dir) / test_run_id
        if run_dir.exists():
            print(f"✓ Created run directory: {run_dir}")

            step_dir = run_dir / "step_100"
            if step_dir.exists():
                print(f"✓ Created step directory: {step_dir}")

                png_files = list(step_dir.glob("*.png"))
                if png_files:
                    print(f"✓ Created {len(png_files)} plot files:")
                    for f in png_files:
                        print(f"    - {f.name}")

                json_file = step_dir / "metrics_history.json"
                if json_file.exists():
                    print(f"✓ Created metrics history: {json_file.name}")

                latest_link = run_dir / "latest"
                if latest_link.exists() and latest_link.is_symlink():
                    print(f"✓ Created latest symlink -> {latest_link.resolve().name}")
        else:
            print("✗ Failed to create run directory")
    else:
        print("AsyncMetricsPlotterCallback is disabled in config")

    print("\n" + "="*60)
    print("Callback testing complete!")
    print("="*60)
    print("\nDirectory structure created:")
    print(f"  completions/{test_run_id}/")
    print(f"    └── step_100/")
    print(f"          └── completions.txt")
    print(f"  plots/{test_run_id}/")
    print(f"    └── step_100/")
    print(f"          ├── train_grad_norm.png")
    print(f"          ├── train_reward_mean.png")
    print(f"          └── metrics_history.json")
    print("\nIf tests passed, your callbacks are configured correctly.")
    print("You can now run the full training with confidence.")


def main():
    parser = argparse.ArgumentParser(description='Test callback functionality')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    test_callbacks(args.config)


if __name__ == "__main__":
    main()
