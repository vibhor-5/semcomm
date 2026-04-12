import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
import torchvision.utils as vutils

def setup_plot(title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

def save_plot(save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=300)
    plt.savefig(save_path + '.pdf', dpi=300)
    plt.close()

def plot_rate_semantic_curve(results: list[dict], baseline_results: list[dict], save_path: str):
    setup_plot('Rate-Semantic Curve', 'Bits per pixel (bpp)', 'CLIP Similarity')
    for res in results:
        plt.plot(res['bpp'], res['clip'], marker='o', label=res['model_name'])
    for res in baseline_results:
        plt.plot(res['bpp'], res['clip'], linestyle='--', marker='x', label=res['model_name'])
    plt.legend()
    save_plot(save_path)

def plot_snr_curve(results: list[dict], save_path: str):
    setup_plot('Channel Robustness', 'SNR (dB)', 'CLIP Similarity')
    for res in results:
        plt.plot(res['snr'], res['clip_snr'], marker='o', label=res['model_name'])
    plt.axhline(y=0.80, color='r', linestyle='--', label='Threshold (0.80)')
    plt.legend()
    save_plot(save_path)

def plot_speed_quality(results: list[dict], save_path: str):
    setup_plot('Speed vs Quality', 'Inference Time (ms/image)', 'CLIP Similarity')
    for res in results:
        plt.scatter(res['ms_per_img'], res['clip'], label=res['model_name'], s=100)
    plt.legend()
    save_plot(save_path)

def plot_reconstruction_gallery(orig: torch.Tensor, recons: dict[str, torch.Tensor],
                                 save_path: str, n_images: int = 8):
    img_list = []
    
    orig_sel = orig[:n_images].cpu()
    img_list.append(orig_sel)
    
    model_names = list(recons.keys())
    for name in model_names:
        img_list.append(recons[name][:n_images].cpu())
        
    grid = torch.cat(img_list, dim=0)
    
    fig, ax = plt.subplots(figsize=(n_images*2, (len(model_names)+1)*2))
    ax.axis("off")
    grid_disp = (grid + 1) / 2
    
    img_grid = vutils.make_grid(grid_disp, nrow=n_images, padding=2, normalize=False)
    ax.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
    
    row_names = ['Original'] + model_names
    for i, name in enumerate(row_names):
        ax.text(-0.02, 1 - (i + 0.5) / len(row_names), name, va='center', ha='right', fontsize=14, transform=ax.transAxes)
        
    save_plot(save_path)

def plot_clip_histogram(clip_scores: dict[str, list[float]], save_path: str):
    setup_plot('CLIP Score Distribution', 'CLIP Similarity', 'Density')
    for model_name, scores in clip_scores.items():
        sns.kdeplot(scores, label=model_name, fill=True, alpha=0.3)
    plt.legend()
    save_plot(save_path)
