"""
Custom callbacks for GRPO training.

Provides logging and visualization callbacks that run asynchronously
to avoid interrupting training.
"""

import json
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for cluster usage
import matplotlib.pyplot as plt
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class CompletionLoggerCallback(TrainerCallback):
    """
    Logs model completions to text files when the model is saved.
    Files are organized by run ID and step number.
    """

    def __init__(self, output_dir: str = "completions", run_id: str = None):
        self.base_dir = Path(output_dir)
        self.run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = self.base_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs):
        """Save completions when model checkpoint is saved."""
        step = state.global_step

        # Create step directory
        step_dir = self.output_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Extract completions
        completions = self._extract_completions(state)

        if completions:
            output_file = step_dir / "completions.txt"
            self._save_completions(completions, output_file, step)

            # Update latest symlink
            self._update_latest_symlink(step_dir)

            print(f"Saved {len(completions)} completions to {output_file}")

    def _update_latest_symlink(self, step_dir: Path):
        """Update the 'latest' symlink to point to the most recent step directory."""
        latest_link = self.output_dir / "latest"

        # Remove old symlink if it exists
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink (relative path for portability)
        latest_link.symlink_to(step_dir.name)

    def _extract_completions(self, state: TrainerState) -> List[Dict[str, str]]:
        """Extract completions from trainer state."""
        completions = []

        # GRPO logs completions in the log history
        # Look for the most recent log entry with completions
        for log_entry in reversed(state.log_history):
            # Try different possible keys where completions might be stored
            possible_keys = [
                'completions',
                'train/completions',
                'eval/completions',
                'generation/completions',
                'samples',
                'train/samples'
            ]

            for key in possible_keys:
                if key in log_entry:
                    completions = log_entry[key]
                    print(f"Found completions under key: {key}")
                    break

            if completions:
                break

        if not completions:
            print("Warning: No completions found in log history. "
                  "Check if GRPO trainer is logging completions correctly.")

        return completions

    def _save_completions(self, completions: List, output_file: Path, step: int):
        """Save completions to a text file."""
        with open(output_file, 'w') as f:
            f.write(f"Completions at Step {step}\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            for i, item in enumerate(completions):
                f.write(f"### Completion {i+1} ###\n")

                # Handle different formats of completion data
                if isinstance(item, dict):
                    if 'prompt' in item:
                        f.write(f"Prompt: {item['prompt']}\n")
                    if 'completion' in item:
                        f.write(f"Completion: {item['completion']}\n")
                    if 'reward' in item:
                        f.write(f"Reward: {item['reward']}\n")
                elif isinstance(item, str):
                    f.write(f"Completion: {item}\n")

                f.write("\n" + "-" * 40 + "\n\n")


class AsyncMetricsPlotterCallback(TrainerCallback):
    """
    Plots training metrics asynchronously without blocking training.
    Collects metrics on every log but only plots on save to avoid overhead.
    Continuously updates plots in the run directory to show full history.
    """

    def __init__(self, plot_dir: str = "plots",
                 metrics_to_plot: Optional[List[str]] = None,
                 plot_on_save: bool = True,
                 run_id: str = None):
        self.base_dir = Path(plot_dir)
        self.run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.plot_dir = self.base_dir / self.run_id
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Update latest symlink to point to this run directory
        self._update_latest_symlink()

        # Default metrics to plot if none specified
        if metrics_to_plot is None:
            metrics_to_plot = [
                "train/grad_norm",
                "train/reward_mean",
                "train/reward_std",
                "train/kl_divergence",
                "learning_rate"
            ]

        self.metrics_to_plot = metrics_to_plot
        self.plot_on_save = plot_on_save
        self.metrics_history = {metric: [] for metric in metrics_to_plot}
        self.steps = []
        self._lock = threading.Lock()

    def _update_latest_symlink(self):
        """Update the 'latest' symlink to point to the current run directory."""
        latest_link = self.base_dir / "latest"

        # Remove old symlink if it exists
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink (relative path for portability)
        latest_link.symlink_to(self.run_id)

    def on_log(self, args: TrainingArguments, state: TrainerState,
              control: TrainerControl, logs: Optional[Dict[str, float]] = None,
              **kwargs):
        """Collect metrics on every log event and optionally plot."""
        if logs is None:
            return

        with self._lock:
            # Record step
            step = state.global_step
            if step > 0:  # Skip step 0 which might have incomplete metrics
                self.steps.append(step)

                # Collect each metric
                for metric_name in self.metrics_to_plot:
                    # Handle different possible metric name formats
                    value = None
                    if metric_name in logs:
                        value = logs[metric_name]
                    elif metric_name.replace('train/', '') in logs:
                        value = logs[metric_name.replace('train/', '')]
                    elif 'train/' + metric_name in logs:
                        value = logs['train/' + metric_name]

                    # Append value (or None if not found)
                    self.metrics_history[metric_name].append(value)

        # Plot on every log if plot_on_save is False
        if not self.plot_on_save and step > 0:
            # Copy data to avoid threading issues
            with self._lock:
                steps_copy = self.steps.copy()
                metrics_copy = {k: v.copy() for k, v in self.metrics_history.items()}

            if len(steps_copy) == 0:
                return

            # Plot asynchronously
            thread = threading.Thread(
                target=self._plot_metrics,
                args=(steps_copy, metrics_copy, state.global_step)
            )
            thread.daemon = True
            thread.start()

    def on_save(self, args: TrainingArguments, state: TrainerState,
                control: TrainerControl, **kwargs):
        """Plot metrics when model is saved (only if plot_on_save is True)."""
        if not self.plot_on_save:
            return

        # Copy data to avoid threading issues
        with self._lock:
            steps_copy = self.steps.copy()
            metrics_copy = {k: v.copy() for k, v in self.metrics_history.items()}

        if len(steps_copy) == 0:
            return

        # Plot asynchronously (plots go directly in run directory)
        thread = threading.Thread(
            target=self._plot_metrics,
            args=(steps_copy, metrics_copy, state.global_step)
        )
        thread.daemon = True
        thread.start()

    def _plot_metrics(self, steps: List[int], metrics: Dict[str, List],
                      current_step: int):
        """Generate and save metric plots."""
        try:
            # Plot each metric separately
            for metric_name in self.metrics_to_plot:
                values = metrics[metric_name]

                # Filter out None values
                valid_points = [(s, v) for s, v in zip(steps, values) if v is not None]

                if not valid_points:
                    print(f"No data for {metric_name}, skipping plot")
                    continue

                # Create individual figure for this metric
                fig, ax = plt.subplots(figsize=(8, 5))

                valid_steps, valid_values = zip(*valid_points)
                ax.plot(valid_steps, valid_values, marker='o', markersize=3,
                       linewidth=2, color='steelblue')
                ax.set_xlabel('Step', fontsize=12)
                ax.set_ylabel(metric_name.replace('train/', '').replace('_', ' ').title(), fontsize=12)
                ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Add current value annotation
                current_value = valid_values[-1]
                ax.annotate(f'Current: {current_value:.6g}',
                          xy=(0.02, 0.98), xycoords='axes fraction',
                          verticalalignment='top', fontsize=11,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue',
                                   edgecolor='steelblue', alpha=0.7))

                # Add step and timestamp info
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ax.text(0.98, 0.02, f'Step {current_step} | {timestamp}',
                       transform=ax.transAxes, fontsize=9, color='gray',
                       horizontalalignment='right', verticalalignment='bottom')

                plt.tight_layout()

                # Save directly to run directory (continuously update)
                metric_filename = metric_name.replace('/', '_').replace(' ', '_')
                plot_file = self.plot_dir / f'{metric_filename}.png'
                plt.savefig(plot_file, dpi=100, bbox_inches='tight')

                plt.close(fig)

            # Save metrics history as JSON
            history_file = self.plot_dir / 'metrics_history.json'
            history_data = {
                'steps': steps,
                'metrics': metrics,
                'current_step': current_step,
                'timestamp': datetime.now().isoformat()
            }
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)

            print(f"Updated {len(self.metrics_to_plot)} plots in {self.plot_dir}/")

        except Exception as e:
            print(f"Error plotting metrics: {e}")
            # Don't crash training due to plotting errors


class CallbackRegistry:
    """Registry for managing callbacks based on configuration."""

    @staticmethod
    def create_callbacks(config: Dict[str, Any], config_path: str = None) -> List[TrainerCallback]:
        """
        Create callbacks based on configuration.

        Args:
            config: Configuration dictionary
            config_path: Path to config file (used to extract config name for run ID)
        """
        callbacks = []
        callback_config = config.get('callbacks', {})

        # Generate run ID from config name and timestamp
        run_id = CallbackRegistry._generate_run_id(config_path)
        print(f"Run ID: {run_id}")

        # Completion Logger
        completion_config = callback_config.get('completion_logger', {})
        if completion_config.get('enabled', False):
            callbacks.append(CompletionLoggerCallback(
                output_dir=completion_config.get('output_dir', 'completions'),
                run_id=run_id
            ))
            output_path = Path(completion_config.get('output_dir', 'completions')) / run_id
            print(f"Enabled CompletionLoggerCallback -> {output_path}")

        # Metrics Plotter
        plotter_config = callback_config.get('metrics_plotter', {})
        if plotter_config.get('enabled', False):
            callbacks.append(AsyncMetricsPlotterCallback(
                plot_dir=plotter_config.get('plot_dir', 'plots'),
                metrics_to_plot=plotter_config.get('metrics_to_plot', None),
                plot_on_save=plotter_config.get('plot_on_save', True),
                run_id=run_id
            ))
            plot_path = Path(plotter_config.get('plot_dir', 'plots')) / run_id
            print(f"Enabled AsyncMetricsPlotterCallback -> {plot_path}")

        return callbacks

    @staticmethod
    def _generate_run_id(config_path: str = None) -> str:
        """
        Generate a run ID from config name and timestamp.

        Format: configname_YYYY-MM-DD_HH-MM-SS
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if config_path:
            # Extract config name from path
            config_name = Path(config_path).stem  # e.g., "llama2-7b-gsm8k" from "configs/llama2-7b-gsm8k.yaml"
            run_id = f"{config_name}_{timestamp}"
        else:
            # Fallback to just timestamp
            run_id = timestamp

        return run_id
