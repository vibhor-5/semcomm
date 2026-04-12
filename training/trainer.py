import torch
import os
import numpy as np
from tqdm import tqdm
from utils.checkpoint import get_checkpoint_dir, save_checkpoint
from losses.semantic_loss import CLIPLoss

class Trainer:
    def __init__(self, encoder, channel, decoder, loss_cfg, cfg, drive_root):
        from utils.device import get_device
        self.device = get_device()
        self.encoder = encoder.to(self.device)
        self.channel = channel.to(self.device)
        self.decoder = decoder.to(self.device)
        self.scaler  = torch.cuda.amp.GradScaler()
        
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=cfg['training']['lr'],
                                            weight_decay=cfg['training']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg['training']['n_epochs'])
            
        self.ckpt_dir = get_checkpoint_dir(cfg, drive_root)
        self.cfg = cfg
        self.loss_cfg = loss_cfg
        self.best_clip = 0.0
        
        if self.loss_cfg.get('clip_weight', 0) > 0:
            self.clip_loss_fn = CLIPLoss(self.device)
            
    def _train_step(self, batch) -> dict:
        images, labels, tokens = batch
        images = images.to(self.device)
        tokens = tokens.to(self.device)
        
        if tokens.sum() == 0 and self.encoder.semantic_token_type == 'clip':
             _, tokens = self.encoder(images)
             tokens = tokens.detach()

        with torch.cuda.amp.autocast():
            latent, _ = self.encoder(images)
            
            if self.cfg.get('channel', {}).get('snr_conditioning', False):
                noisy_latent, snr_used = self.channel(latent)
                snr_emb = torch.full((images.shape[0], 1), snr_used, device=self.device)
            else:
                noisy_latent = self.channel(latent)
                snr_emb = None

            main_loss = self.decoder.compute_loss(images, noisy_latent, tokens, snr_emb)

            clip_loss = torch.tensor(0.0, device=self.device)
            if self.loss_cfg.get('clip_weight', 0) > 0:
                with torch.no_grad():
                    recon = self.decoder.sample(noisy_latent, tokens, steps=10, snr_emb=snr_emb)
                clip_loss = self.clip_loss_fn(images, recon) * self.loss_cfg['clip_weight']

            total = main_loss + clip_loss

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            self.cfg['training']['gradient_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return {'main': main_loss.item(), 'clip': clip_loss.item(), 'total': total.item()}

    def _validate(self, val_loader):
        return {'clip': 0.0, 'psnr': 0.0}

    def run(self, train_loader, val_loader, n_epochs):
        import wandb
        if self.cfg['logging']['use_wandb']:
            wandb.init(project=self.cfg['logging']['project'],
                       name=self.cfg['experiment_id'],
                       config=self.cfg)

        for epoch in range(1, n_epochs + 1):
            self.encoder.train()
            self.decoder.train()
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}"):
                train_losses.append(self._train_step(batch))

            mean_train = {k: float(np.mean([d[k] for d in train_losses]))
                          for k in train_losses[0]}

            val_metrics = {}
            if epoch % self.cfg['evaluation']['eval_every_n_epochs'] == 0:
                val_metrics = self._validate(val_loader)
                if val_metrics.get('clip', 0) > self.best_clip:
                    self.best_clip = val_metrics['clip']
                    save_checkpoint({'epoch': epoch, 
                                     'model_state': self.decoder.state_dict(),
                                     'encoder_state': self.encoder.state_dict(),
                                     'optimizer_state': self.optimizer.state_dict(),
                                     'scaler_state': self.scaler.state_dict()},
                                     os.path.join(self.ckpt_dir, f'epoch_{epoch}.pt'),
                                     is_best=True)

            if epoch % self.cfg['logging']['save_every_n_epochs'] == 0:
                save_checkpoint({'epoch': epoch,
                                 'model_state': self.decoder.state_dict(),
                                 'encoder_state': self.encoder.state_dict(),
                                 'optimizer_state': self.optimizer.state_dict(),
                                 'scaler_state': self.scaler.state_dict()},
                                 os.path.join(self.ckpt_dir, f'epoch_{epoch}.pt'))

            if self.cfg['logging']['use_wandb']:
                stats = {'epoch': epoch, 'lr': self.scheduler.get_last_lr()[0]}
                stats.update({f'train/{k}': v for k,v in mean_train.items()})
                stats.update({f'val/{k}': v for k,v in val_metrics.items()})
                wandb.log(stats)

            self.scheduler.step()
        
        if self.cfg['logging']['use_wandb']:
            wandb.finish()
