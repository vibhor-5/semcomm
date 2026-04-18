"""
train_flow.py — Entry point for Group B (Flow-Matching) experiments.

Usage:
    python training/train_flow.py --config configs/flow/B1_linear.yaml \
           --data_dir /content/drive/MyDrive/semcomm/data \
           --output_dir /content/drive/MyDrive/semcomm/outputs
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_experiment import main

if __name__ == '__main__':
    # Inject a helper flag so downstream callers know we're in flow-only mode
    # (the generic run_experiment.py handles both flow and diffusion, so we just call it)
    main()
