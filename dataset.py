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

class RadioDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path:str,
                 config:dict,
                 sample_rate=16000) -> None:
        super().__init__()

        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx:int) -> Tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]

        with torch.no_grad():
            source, sr = torchaudio.load(row['clean'])
            std, mean = torch.std_mean(source, dim=-1)
            source = (source - mean)/std

            mixture, _ = torchaudio.load(row['noisy'])
            std, mean = torch.std_mean(mixture, dim=-1)
            mixture = (mixture - mean)/std
        
        return torch.t(mixture), torch.t(source)

def data_processing(data:Tuple[Tensor,Tensor]) -> Tuple[Tensor, Tensor, list]:
    mixtures = []
    sources = []
    lengths = []

    for mixture, source in data:
        # w/o channel
        mixtures.append(mixture)
        sources.append(source)
        lengths.append(len(mixture))

    mixtures = nn.utils.rnn.pad_sequence(mixtures, batch_first=True)
    sources = nn.utils.rnn.pad_sequence(sources, batch_first=True)

    mixtures = mixtures.squeeze()
    sources = sources.squeeze()

    if mixtures.dim() == 1:
        mixtures = mixtures.unsqueeze(0)
        sources = sources.unsqueeze(0)
        
    return mixtures, sources, lengths
