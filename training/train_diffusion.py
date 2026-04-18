"""
train_diffusion.py — Entry point for Group C (Diffusion) experiments.

Usage:
    python training/train_diffusion.py --config configs/diffusion/C1_cross_attention.yaml \
           --data_dir /content/drive/MyDrive/semcomm/data \
           --output_dir /content/drive/MyDrive/semcomm/outputs
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.run_experiment import main

if __name__ == '__main__':
    main()
