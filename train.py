import warnings
warnings.simplefilter("ignore", UserWarning)

from argparse import ArgumentParser
from multiprocessing import Pool
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.utils.data as data
import torchaudio
import numpy as np

from dataset import LibriRepresentationDataset
from models.lightning_model import RepresentationModel

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_root', type=str, default='/home/n1900235d/INTERSPEECH/final_repr_data')
    parser.add_argument('--wav_len', type=int, default=16000 * 2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=float, default=512)
    parser.add_argument('--gpu', type=int, default="4")
    parser.add_argument('--nworkers', type=int, default=2)
    # int(int(Pool()._processes)*0.75))
    parser.add_argument('--dev', type=str, default=False)


    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    print(f'Available : #Cores = {str(int(Pool()._processes))}\t#GPU = {torch.cuda.device_count()}')
    print(f'Used :      #Cores = {hparams.nworkers}\t#GPU = {hparams.gpu}')


    HPARAMS = {
        'data_batch_size' : hparams.batch_size,
        'data_wav_len' : hparams.wav_len,
        'training_lr' : 1e-3,
        'hidden_dim': hparams.hidden_dim,
    }

    train_dataset = LibriRepresentationDataset(root=hparams.data_root, wav_len=hparams.wav_len)

    trainloader = data.DataLoader(
        train_dataset, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=True, 
        num_workers=hparams.nworkers
    )

    model = RepresentationModel(HPARAMS)

    checkpoint_callback = ModelCheckpoint(
        monitor='epoch_loss', 
        mode='min',
        verbose=1)

    trainer = pl.Trainer(fast_dev_run=hparams.dev, 
                        gpus=hparams.gpu, 
                        max_epochs=hparams.epochs, 
                        checkpoint_callback=checkpoint_callback,
                        distributed_backend='ddp',
                        # logger=logger,
                        # resume_from_checkpoint='/home/shangeth/INTERSPEECH/lightning_logs/version_49/checkpoints/epoch=37.ckpt'
                        )

    trainer.fit(model, train_dataloader=trainloader)
    # trainer.test(model, test_dataloaders=testloader)