"""
Reward functions for GRPO training.

Add new reward functions here and register them in REWARD_REGISTRY.
Enable/disable rewards via the config YAML without modifying this file.
"""

import re


def extract_answer(text):
    """Extract numerical answer from model output."""
    # First, try to find text after the first occurrence of ####
    if "####" in text:
        # Find the first occurrence and get text after it
        idx = text.find("####")
        after_hashes = text[idx + 4:]

        # Extract all numbers from the text after ####
        numbers = re.findall(r"\d+", after_hashes)
        if numbers:
            # Return the first number found after ####
            return int(numbers[0])

    # If no #### found, extract the last number in the entire text
    all_numbers = re.findall(r"\d+", text)
    if all_numbers:
        return int(all_numbers[-1])

    return None


def accuracy_reward(prompts, completions, numerical_answer, **kwargs):
    """Reward correct numerical answers."""
    rewards = []
    for completion, gt in zip(completions, numerical_answer):
        predicted = extract_answer(completion)
        if predicted == gt:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
    return rewards


def format_reward(prompts, completions, **kwargs):
    """Reward proper formatting with #### delimiter."""
    rewards = []
    for completion in completions:
        # Reward for having #### in the output
        if "####" in completion:
            rewards.append(0.2)
        else:
            rewards.append(-0.2)
    return rewards


def length_reward(prompts, completions, **kwargs):
    """Penalize excessively short completions."""
    rewards = []
    for completion in completions:
        # Penalize excessive brevity
        if len(completion) < 30:
            rewards.append(-0.5)
        else:
            # No penalty for longer completions
            rewards.append(0.0)
    return rewards


def step_by_step_reward(prompts, completions, **kwargs):
    """Reward step-by-step reasoning indicators."""
    rewards = []
    for completion in completions:
        score = 0.0
        # Check for step-by-step indicators
        if "step" in completion.lower():
            score += 0.1
        if "first" in completion.lower() or "then" in completion.lower():
            score += 0.1
        if "\n" in completion:  # Multi-line reasoning
            score += 0.1
        rewards.append(score)
    return rewards


def verbosity_penalty(prompts, completions, **kwargs):
    """Penalize overly verbose responses."""
    rewards = []
    for completion in completions:
        word_count = len(completion.split())
        if word_count > 300:
            # Logarithmic penalty for excessive length
            penalty = -0.1 * (word_count - 300) / 100
            rewards.append(max(penalty, -1.0))
        else:
            rewards.append(0.0)
    return rewards


# Registry of all available reward functions
# Add new reward functions here - they will be automatically available
REWARD_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "length": length_reward,
    "step_by_step": step_by_step_reward,
    "verbosity_penalty": verbosity_penalty,
}


def get_reward_functions(reward_names):
    """
    Get reward functions by name from the registry.

    Args:
        reward_names: List of reward function names to use

    Returns:
        List of reward function callables
    """
    reward_funcs = []
    for name in reward_names:
        if name not in REWARD_REGISTRY:
            raise ValueError(f"Reward function '{name}' not found in registry. "
                           f"Available: {list(REWARD_REGISTRY.keys())}")
        reward_funcs.append(REWARD_REGISTRY[name])
    return reward_funcs
