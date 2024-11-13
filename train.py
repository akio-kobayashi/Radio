import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
import torch.utils.data as data
from solver import LitDenoiser
import torch.utils.data as dat
import dataset as rd
from dataset import RadioDataset
from argparse import ArgumentParser
import yaml
import warnings
warnings.filterwarnings('ignore')

def main(config:dict, args):

    model = LitDenoiser(config)
    model.model.to('cuda')
   
    train_dataset = RadioDataset(config['dataset']['train']['csv_path'],
                                 config,
                                 rp_src=args.src_dir,
                                 rp_tgt=args.tgt_dir
                                 )
    train_loader = data.DataLoader(dataset=train_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=True, 
                                   collate_fn=lambda x: rd.data_processing(x))
    valid_dataset = RadioDataset(config['dataset']['valid']['csv_path'],
                                 config,
                                 rp_src=args.src_dir,
                                 rp_tgt=args.tgt_dir
                                 )
    valid_loader = data.DataLoader(dataset=valid_dataset,
                                   **config['dataset']['process'],
                                   pin_memory=True,
                                   shuffle=False, 
                                   collate_fn=lambda x: rd.data_processing(x))
    callbacks = [
        pl.callbacks.ModelCheckpoint( **config['checkpoint'])
    ]
    logger = TensorBoardLogger(**config['logger'])
    trainer = pl.Trainer( callbacks=callbacks,
                          logger=logger,
                          **config['trainer'] )
    trainer.fit(model=model, ckpt_path=args.checkpoint, train_dataloaders=train_loader,
                val_dataloaders=valid_loader)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)
    
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--src_dir', type=str, default=None)
    parser.add_argument('--tgt_dir', type=str, default=None)
    #parser.add_argument('--gpus', nargs='*', type=int, default=0)
    args=parser.parse_args()

    torch.set_float32_matmul_precision('high')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    with open(args.config, 'r') as yf:
        config = yaml.safe_load(yf)
    if 'config' in config.keys():
        config = config['config']
    main(config, args)
