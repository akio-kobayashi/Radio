import argparse
import json
import logging
import sys, os
import torch
import torch.nn.functional as F
import torchaudio
import lightning.pytorch as pl
from solver import LitDenoiser
from argparse import ArgumentParser
import pandas as pd
import yaml
import numpy as np
from einops import rearrange

def read_audio(path):
    wave, sr = torchaudio.load(path)
    return wave

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    #decoder = whisper.load_model("large").to(device)

    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    config = config['config']
    
    assert args.checkpoint is not None
    model = LitDenoiser.load_from_checkpoint(args.checkpoint,
                                             config=config).to(device)
    model.eval()
    
    sample_rate = config['dataset']['segment']['sample_rate']
    df = pd.read_csv(args.input_csv)
    df_out = pd.DataFrame(index=None,
                          columns=['key', 'clean', 'noisy', 'denoise'])
    keys, noisy, clean, denoise = [], [], [], []
    with torch.no_grad():
        for [index, row] in df.iterrows():

            key = row['key']
            keys.append(key)
            
            noisy.append(row['noisy'])
            clean.append(row['clean'])

            noisy_path = row['noisy'].replace(args.src_dir, args.dst_dir)
            clean_path = row['clean'].replace(args.src_dir, args.dst_dir)
            noisy_wav = read_audio(noisy_path)
            clean_wav = read_audio(clean_path)
            
            noisy_original_length = noisy_wav.shape[-1]
            clean_original_length = clean_wav.shape[-1]
            assert noisy_original_length == clean_original_length

            # normalize and padding
            noisy_std, noisy_mean = torch.std_mean(noisy_wav)
            noisy_wav = (noisy_wav - noisy_mean)/noisy_std

            denoise_wav = rearrange(model(noisy_wav.to(device)), 'b c t -> (b c) t')
            denoise_length = min(noisy_original_length, denoise_wav.shape[-1])
            denoise_wav = denoise_wav[:, :denoise_length]
            denoise_wav *= noisy_std
            
            denoise_outdir = args.output_dir
            if not os.path.exists(denoise_outdir):
                os.path.mkdir(denoise_outdir)
            outpath = os.path.join(denoise_outdir, key) + '_denoise.wav'
            torchaudio.save(uri=outpath, src=denoise_wav.to('cpu'),
                            sample_rate=sample_rate)
            denoise.append(outpath)

    df_out['key'], df_out['noisy'], df_out['clean'], df_out['denoise'] = keys, noisy, clean, denoise
    df_out.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--input_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--dst_dir', type=str)
    
    args = parser.parse_args()

    main(args)
