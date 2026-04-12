import os
import torch

def save_checkpoint(state: dict, path: str, is_best: bool = False):
    """Save checkpoint. Optionally also copy to best.pt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best.pt')
        torch.save(state, best_path)
    print(f"Saved checkpoint → {path}")

def load_checkpoint(path: str, model, optimizer=None, scaler=None):
    """Load checkpoint. Returns epoch and best_metric."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    if scaler and 'scaler_state' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state'])
    return ckpt.get('epoch', 0), ckpt.get('best_metric', 0.0)

def get_checkpoint_dir(cfg: dict, drive_root: str) -> str:
    """Returns path like /content/drive/MyDrive/semcomm/outputs/B1_linear/checkpoints/"""
    exp_id = cfg['experiment_id']
    return os.path.join(drive_root, 'outputs', exp_id, 'checkpoints')
