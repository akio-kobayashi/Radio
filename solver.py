import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from demcus import Demucs
from stft_loss import MultiResolutionSTFTLoss
from percept_loss import PerceptualLoss
from typing import Tuple
import os,sys
from einops import rearrange

class HuberLoss(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta, reduction='sum')
        
    def forward(self, preds, targets, lengths):
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :lengths[b]] = 1
        return self.loss(preds*mask, targets*mask)/torch.sum(mask)
    
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='sum')
        
    def forward(self, preds, targets, lengths):
        mask = torch.zeros_like(preds, dtype=preds.dtype, device=preds.device)
        for b in range(len(preds)):
            mask[b, :lengths[b]] = 1
        return self.loss(preds*mask, targets*mask)/torch.sum(mask)
    
class LitDenoiser(pl.LightningModule):
    def __init__(self, config:dict) -> None:
        super().__init__()
        self.config = config
        self.model = Demucs(config['demucs'])

        self.stft_loss = self.percept_loss = None
        if config['loss']['main'] == 'stft_loss':
            self.stft_loss = MultiResolutionSTFTLoss()
        elif config['loss']['main'] == 'percept_loss':
            self.percept_loss = PerceptualLoss()
        self.stft_loss_weight = config['loss']['stft']['weight']
        self.percept_loss_weight = config['loss']['perceptual']['weight']

        self.loss == None
        if config['loss']['type'] == 'huber':
            self.loss = HuberLoss(delta=config['loss']['huber_loss']['delta'])
            self.loss_weight = config['loss']['huber_loss']['weight']
        elif config['loss']['type'] == 'l1_loss':
            self.loss = L1Loss()
            self.loss_weight = config['loss']['l1_loss']['weight']

        self.valid_step_loss=[]
        self.valid_epoch_loss=[]
        
        self.save_hyperparameters()

    def forward(self, mix:Tensor) -> Tensor:
        return self.model(mix)

    def compute_loss(self, estimates, targets, lengths, valid=False)->Tuple[Tensor, dict]:
        prefix='valid_' if valid is True else ''
        
        d = {}
        _loss = 0.
        
        if self.loss is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _loss = self.loss(estimates, targets, lengths)
                d[prefix + 'amp_loss'] = _loss
                _loss = self.loss_weight * _loss

        if self.stft_loss is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _stft_loss1, _stft_loss2 = self.stft_loss(estimates, targets)
                _loss += self.stft_loss_weight * (_stft_loss1 + _stft_loss2)
                d[prefix+'stft_loss'] = _stft_loss1 + _stft_loss2

        if self.percept_loss is not None:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                _percept_loss = self.percept_loss(estimates, targets, lengths)
                _loss += self.percept_loss_weight * _percept_loss
                d[prefix+'percept_loss'] = _percept_loss
                
        if valid is True:
            d['valid_loss'] = _loss
        else:
            d['train_loss'] = _loss
            
        return _loss, d
    
    def training_step(self, batch, batch_idx:int) -> Tensor:
        mixtures, sources, lengths = batch
        sources = rearrange(sources, 'b c t -> (b c) t')

        src_hat = self.forward(mixtures)
        src_hat = rearrange(src_hat, 'b c t -> (b c) t')
        _loss, d = self.compute_loss(src_hat, sources, lengths, valid=False)
        self.log_dict(d)

        return _loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        mixtures, sources, lengths = batch
        sources = rearrange(sources, 'b c t -> (b c) t')

        src_hat = self.forward(mixtures)
        src_hat = rearrange(src_hat, 'b c t -> (b c) t')
        _loss, d = self.compute_loss(src_hat, sources, lengths, valid=True)
        self.log_dict(d)
        self.valid_step_loss.append(_loss.item())
        
        return _loss

    def on_validation_epoch_end(self):
        _loss = np.mean(self.valid_step_loss)
        self.valid_epoch_loss.append(_loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     **self.config['optimizer'])
        return (
            {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100),
                "monitor": "val_loss"
                }
            }
            )
    
    def get_model(self):
        return self.model
    
