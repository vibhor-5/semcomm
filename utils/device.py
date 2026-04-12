import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Apple Silicon fallback (if ever running locally)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def print_device_info():
    dev = get_device()
    print(f"Device: {dev}")
    if dev.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
