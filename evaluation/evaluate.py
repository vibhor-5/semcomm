import os
import json
import torch
from tqdm import tqdm
from .metrics import (
    compute_clip_similarity, compute_lpips, compute_fid,
    compute_psnr, compute_ssim, compute_bpp
)
from utils.checkpoint import load_checkpoint

def run_evaluation(cfg, checkpoint_path, test_loader, device, encoder, channel, decoder):
    import clip
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()
        
    load_checkpoint(checkpoint_path, decoder, optimizer=None, scaler=None)
    # the encoder might share the checkpoint or load separately if needed.
    # In guide: load_checkpoint(checkpoint_path, decoder)  # encoder weights also in ckpt
    
    encoder.eval()
    decoder.eval()

    all_orig, all_recon = [], []
    with torch.no_grad():
        for images, _, tokens in tqdm(test_loader, desc='Evaluating'):
            images, tokens = images.to(device), tokens.to(device)
            latent, _ = encoder(images)
            
            if cfg.get('channel', {}).get('snr_conditioning', False):
                noisy, snr_used = channel(latent)
                snr_emb = torch.full((images.shape[0], 1), snr_used, device=device)
            else:
                noisy = channel(latent)
                snr_emb = None
                
            recon = decoder.sample(noisy, tokens,
                                    steps=cfg['evaluation'].get('sampling_steps', 50),
                                    guidance_scale=cfg['evaluation'].get('guidance_scale', 1.0),
                                    snr_emb=snr_emb)
            all_orig.append(images.cpu())
            all_recon.append(recon.cpu())

    orig  = torch.cat(all_orig)
    recon = torch.cat(all_recon)

    metrics = {
        'clip':  compute_clip_similarity(orig, recon, clip_model, device),
        'lpips': compute_lpips(orig, recon, device),
        'fid':   compute_fid(orig, recon[:cfg['evaluation']['n_fid_samples']], device),
        'psnr':  compute_psnr(orig, recon),
        'ssim':  compute_ssim(orig, recon),
        'bpp':   compute_bpp(cfg['encoder']['latent_dim'],
                              cfg['encoder'].get('quant_bits', 8),
                              cfg['dataset']['image_size'],
                              cfg['dataset']['image_size']),
    }
    
    output_dir = os.path.dirname(os.path.dirname(checkpoint_path)) 
    save_path = os.path.join(output_dir, 'metrics.json')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump({**metrics, 'experiment_id': cfg['experiment_id']}, f, indent=2)

    return metrics
