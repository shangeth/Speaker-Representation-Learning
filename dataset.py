import warnings
warnings.simplefilter("ignore", UserWarning)

from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
import random

class LibriSRDataset(Dataset):
    def __init__(self, root, url, 
                    wav_len=16000, 
                    train=True,
                    noisedir='/home/shangeth/speaker_profiling/noise_datadir/noises'):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url)
        self.labels = sorted([int(x) for x in os.listdir(os.path.join(root, 'LibriSpeech/train-clean-100/'))])
        self.label_dict = {k: v for v, k in enumerate(self.labels)}
        # print(self.label_dict)

        # Transforms
        self.wav_len = wav_len
        self.train_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random')
            ])

        self.test_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len)
            ])

        self.train = train
        self.noisedir = noisedir
        if self.noisedir:
            self.noise_transform = wavencoder.transforms.AdditiveNoise(self.noisedir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, _, _, y, _, _ = self.dataset[idx]
        if self.train:
            x = self.train_transform(x)
            if random.random() < 0.5:
                x = self.noise_transform(x)
        else:
            x = self.test_transform(x)

        y = self.label_dict[y]
        if type(x).__module__ == np.__name__:
            x = torch.tensor(x)

        return x, y


class LibriMIDataset(Dataset):
    def __init__(self, root,
                    wav_len=8000, 
                    train=True):
        self.root = root
        self.speakers = os.listdir(root)

        # Transforms
        self.wav_len = wav_len
        self.train_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random')
            ])
        self.spectral_transform = torchaudio.transforms.MFCC(log_mels=True)

    def __len__(self):
        return 25600

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Anchor
        query_speaker = random.choice(self.speakers)
        filename = random.choice(os.listdir(os.path.join(self.root, query_speaker)))
        x, _ = torchaudio.load(os.path.join(self.root, query_speaker, filename))
        x = self.train_transform(x)

        # Positive 
        if random.random()>0.5:
            xp, _ = torchaudio.load(os.path.join(self.root, query_speaker, filename))
            xp = self.train_transform(xp)
        else:
            p_key_speaker = query_speaker
            filename = random.choice(os.listdir(os.path.join(self.root, p_key_speaker)))
            xp, _ = torchaudio.load(os.path.join(self.root, p_key_speaker, filename))
            xp = self.train_transform(xp)

        # Negative 
        n_key_speaker = random.choice(list(set(self.speakers) - set([query_speaker])))
        filename = random.choice(os.listdir(os.path.join(self.root, n_key_speaker)))
        xn, _ = torchaudio.load(os.path.join(self.root, n_key_speaker, filename))
        xn = self.train_transform(xn)

        x = self.spectral_transform(x)
        xp = self.spectral_transform(xp)
        xn = self.spectral_transform(xn)
        
        return x, xp, xn


class EmbDataset(Dataset):
    def __init__(self, root,
                    wav_len=8000):
        self.root = root
        self.files = os.listdir(root)
        self.info_file = '/home/shangeth/INTERSPEECH/LibriSpeech/SPEAKERS.TXT'
        df = pd.read_csv(self.info_file, skiprows=11, delimiter='|', error_bad_lines=False)
        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.assign(
                sex=df['sex'].apply(lambda x: x.strip()),
                subset=df['subset'].apply(lambda x: x.strip()),
                name=df['name'].apply(lambda x: x.strip()),
            )
        self.info_df = df
        self.spectral_transform = torchaudio.transforms.MFCC(log_mels=True)
        # print(self.info_df.head())

        # Transforms
        self.wav_len = wav_len
        self.transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len)
            ])


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Anchor
        filename = self.files[idx]
        ys = int(filename.split('-')[0])
        yg = self.info_df.loc[self.info_df['id'] == ys]['sex'].values[0]

        x, _ = torchaudio.load(os.path.join(self.root, filename))
        x = self.transform(x)
        x = self.spectral_transform(x)
        return x, ys, yg