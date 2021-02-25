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

from dataset import LibriMIDataset
from models.PLModel import RepresentationModel

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_root', type=str, default='./final_repr_data')
    parser.add_argument('--wav_len', type=int, default=16000 * 3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--hidden_dim', type=float, default=128)
    parser.add_argument('--out_dim', type=float, default=251)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--nworkers', type=int, default=int(int(Pool()._processes)*0.75))
    parser.add_argument('--dev', type=str, default=False)


    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    print(f'#Cores = {hparams.nworkers}\t#GPU = {hparams.gpu}')


    HPARAMS = {
        'data_path' : hparams.data_root,
        'data_wav_len' : hparams.wav_len,
        'data_batch_size' : hparams.batch_size,
        'data_wav_augmentation' : 'Random Crop, Additive Noise',
        'data_label_scale' : 'Standardization',

        'training_optimizer' : 'Adam',
        'training_lr' : 1e-3,
        'training_lr_scheduler' : '-',

        'model_architecture' : 'wav2vec + soft-attention',
        'model_finetune' : 'Layer 5&6',
        'hidden_dim': hparams.hidden_dim,
        'out_dim': hparams.out_dim,
    }

    train_dataset = LibriMIDataset(root=hparams.data_root, wav_len=hparams.wav_len)

    trainloader = data.DataLoader(
        train_dataset, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=True, 
        num_workers=hparams.nworkers
    )

    model = RepresentationModel(HPARAMS)
    # print('Model = ', model)

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
                        resume_from_checkpoint='/home/shangeth/INTERSPEECH/lightning_logs/version_49/checkpoints/epoch=37.ckpt'
                        )

    trainer.fit(model, train_dataloader=trainloader)
    # trainer.test(model, test_dataloaders=testloader)