from transformers import AutoModel
import torch
import torch.nn as nn
from einops import rearrange
from argparse import ArgumentParser
import numpy as np

class PerceptualLoss(nn.Module):
    model_id = "facebook/wav2vec2-xls-r-300m"

    def __init__(self, config):        
        super().__init__()
        self.model = AutoModel.from_pretrained(PerceptualLoss.model_id)
        for param in self.model.parameters():
            param.requires_grad = False
        ##self.downsample_ratio = self.model.config.conv_stride[-1]

        self.ploss_type = config['loss']['ploss_type'] # l1 or l2
        self.loss = None
        if self.ploss_type == 'l1':
            self.loss = nn.L1Loss(reduction='sum')
        else:
            self.loss = nn.MSELoss(reduction='sum')
            
    def forward(self, y_pred, y_true, lengths):
        valid_lengths = [int(np.floor(length/320)) for length in lengths]
        y_pred = self.model(y_pred, output_hidden_states=True)
        y_true = self.model(y_true, output_hidden_states=True)
        valid_lengths = [min(length//320, (y_pred.hidden_states[0]).shape[1]) for length in lengths]

        _loss = 0.
        for p, t in zip(y_pred.hidden_states, y_true.hidden_states):
            # shape = (B, T, F)
            mask = torch.zeros_like(p, dtype=p.dtype, device=p.device)
            for b in range(p.shape[0]):
                #print(valid_lengths[b], p.shape)
                mask[b, :valid_lengths[b], :] = 1
            _loss += self.loss(p*mask, t*mask) / torch.sum(mask)
            
        return _loss
    
