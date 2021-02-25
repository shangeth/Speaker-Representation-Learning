from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchaudio
import numpy as np
from tqdm import tqdm

from dataset import EmbDataset
from models.PLModel import RepresentationModel

from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_root', type=str, default='./final_emb_data')
    parser.add_argument('--wav_len', type=int, default=48000)
    parser.add_argument('--model_path', type=str, default="lightning_logs/version_51/checkpoints/epoch=1422.ckpt")
    parser.add_argument('--hidden_dim', type=float, default=128)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    HPARAMS = {
        'data_path' : hparams.data_root,
        'data_wav_len' : hparams.wav_len,
        'data_wav_augmentation' : 'Random Crop, Additive Noise',
        'data_label_scale' : 'Standardization',

        'training_optimizer' : 'Adam',
        'training_lr' : 1e-3,
        'training_lr_scheduler' : '-',

        'model_architecture' : 'wav2vec + soft-attention',
        'model_finetune' : 'Layer 5&6',
        'hidden_dim': hparams.hidden_dim,
    }

    test_dataset = EmbDataset(root=hparams.data_root, wav_len=hparams.wav_len)
    # model = RepresentationModel.load_from_checkpoint(hparams.model_path, hparams=HPARAMS)
    model = RepresentationModel(HPARAMS)
    checkpoint = torch.load(hparams.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])

    # x, ys, yg = test_dataset[0]
    # print(x.shape, 'S=' ,ys,'G=', yg)

    embs = []
    labels = []
    gender = []

    for x, ys, yg in tqdm(test_dataset):
        # z = torch.randn(1, 100).view(-1)
        z = model(x.unsqueeze(0))
        z_a = model.A(z).view(-1).cpu()
        embs.append(z_a)

        labels.append(ys)
        gender.append(yg)

    embs = torch.stack(embs)
    writer = SummaryWriter()
   
    writer.add_embedding(embs, metadata=gender, tag='gender')
    writer.add_embedding(embs, metadata=labels, tag='speakers')
    writer.close()


    