import numpy as np
import sys, os, re, gzip, struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
from typing import Tuple
from torch import Tensor
import torchaudio
from einops import rearrange

class RadioDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path:str,
                 config:dict,
                 sample_rate=16000,
                 rp_src=None,
                 rp_tgt=None) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.rp_src = rp_src
        self.rp_tgt = rp_tgt
        self.segment = config['dataset']['segment']['segment']
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]

        with torch.no_grad():
            clean_path = row['clean']
            noisy_path = row['noisy']
            if self.rp_src:
                clean_path = clean_path.replace(self.rp_src, self.rp_tgt)
                noisy_path = noisy_path.replace(self.rp_src, self.rp_tgt)
                
            source, sr = torchaudio.load(clean_path)
            std, mean = torch.std_mean(source, dim=-1)
            source = (source - mean)/std

            mixture, _ = torchaudio.load(noisy_path)
            std, mean = torch.std_mean(mixture, dim=-1)
            mixture = (mixture - mean)/std

            # speech tensor shape = (1, T)
            if self.segment > 0:
                max_length = self.segment * self.sample_rate
                _, T = mixture.shape
                if T > max_length:
                    start = np.random.randint(0, T - max_length+1)
                    source = source[:, start:start+max_length]
                    mixture = mixture[:, start:start+max_length]

        return torch.t(mixture), torch.t(source)

def data_processing(data:Tuple[Tensor,Tensor]) -> Tuple[Tensor, Tensor, list]:
    mixtures = []
    sources = []
    lengths = []

    for mixture, source in data:
        # w/o channel (T, 1)
        mixtures.append(mixture)
        sources.append(source)
        lengths.append(len(mixture))

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)

    mixtures = mixtures.squeeze()
    sources = sources.squeeze()

    if mixtures.dim() == 2:
        mixtures = mixtures.unsqueeze(-1)
        sources = sources.unsqueeze(-1)

    mixtures = rearrange(mixtures, 'b t c -> b c t')
    sources = rearrange(sources, 'b t c -> b c t')
    
    return mixtures, sources, lengths

if __name__ == '__main__':
    from argparse import ArgumentParser
    import yaml
    
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--src_dir', type=str, default=None)
    parser.add_argument('--tgt_dir', type=str, default=None)
    args=parser.parse_args()

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)

    csv_path = config['dataset']['train']['csv_path']
    dataset = RadioDataset(csv_path, config, rp_src=args.src_dir, rp_tgt=args.tgt_dir)
    print(len(dataset))
    mix, clean = dataset.__getitem__(0)
    print(mix.shape)
    print(clean.shape)
