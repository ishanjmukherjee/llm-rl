prompt="Read wandb/latest-run plots, files/media (for reasoning traces, which can be 
  long), and run-*.wandb (and run python tools/wand_dump.py) to get the text 
  version. Interpret current training progress."

claude -p "$prompt" --permission-mode bypassPermissions --verbose --model sonnet --dangerously-skip-permissions --output-format stream-json