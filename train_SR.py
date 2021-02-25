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

from dataset import LibriSRDataset
from models.PLModel import SpeakerRecognitionModel

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--wav_len', type=int, default=16000*5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=float, default=128)
    parser.add_argument('--out_dim', type=float, default=251)
    parser.add_argument('--gpu', type=int, default="1")
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

    train_dataset = LibriSRDataset(root=hparams.data_root, url="train-clean-100", wav_len=hparams.wav_len)
    dev_dataset = LibriSRDataset(root=hparams.data_root, url="dev-clean", wav_len=hparams.wav_len, train=False)
    test_dataset = LibriSRDataset(root=hparams.data_root, url="test-clean", wav_len=hparams.wav_len, train=False)

    trainloader = data.DataLoader(
        train_dataset, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=True, 
        num_workers=hparams.nworkers
    )

    devloader = data.DataLoader(
        dev_dataset, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.nworkers
    )

    testloader = data.DataLoader(
        test_dataset, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.nworkers
    )

    labels1 = os.listdir(f'{hparams.data_root}LibriSpeech/train-clean-100/')
    labels2 = os.listdir(f'{hparams.data_root}LibriSpeech/dev-clean/')

    print(len(train_dataset), len(dev_dataset), len(test_dataset))
    print(len(labels1), len(labels2))
    print(np.setdiff1d(labels2,labels1))

    # model = SpeakerRecognitionModel(HPARAMS)

    # checkpoint_callback = ModelCheckpoint(
    #     monitor='v_loss', 
    #     mode='min',
    #     verbose=1)

    # trainer = pl.Trainer(fast_dev_run=hparams.dev, 
    #                     gpus=hparams.gpu, 
    #                     max_epochs=hparams.epochs, 
    #                     checkpoint_callback=checkpoint_callback,
    #                     # logger=logger,
    #                     # resume_from_checkpoint='/home/shangeth/NISP/NISP_logs/version_1/checkpoints/epoch=196.ckpt'
    #                     )

    # trainer.fit(model, train_dataloader=trainloader, val_dataloaders=devloader)
    # trainer.test(model, test_dataloaders=testloader)