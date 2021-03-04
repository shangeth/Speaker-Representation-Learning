import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
# from models.encoder import Encoder2, Discriminator, Accumulator
from models.model import Encoder, Accumulator, CenterLoss, Discriminator

# from wavencoder.models import Wav2Vec
import torch_optimizer as optim

class RepresentationModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        self.save_hyperparameters()

        self.E = Encoder(HPARAMS['hidden_dim'])
        self.A = Accumulator(HPARAMS['hidden_dim'])
        self.D = Discriminator(2*HPARAMS['hidden_dim'])

        self.intra_utter_criterion = CenterLoss()
        self.classification_criterion = nn.BCELoss()
        self.lr = HPARAMS['training_lr']

        print(self.E)
        print(self.A)
        print(self.D)
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.E(x)
        return x
        
    def configure_optimizers(self):
        optimizer = optim.DiffGrad(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        # waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
        x, xp, xn = batch
        z = self(x)
        zp = self(xp)
        zn = self(xn)

        z_a = self.A(z)
        zp_a = self.A(zp)
        zn_a = self.A(zn)

        yp = self.D(z_a, zp_a)
        yn = self.D(z_a, zn_a)

        loss_p = self.classification_criterion(yp, torch.ones_like(yp, device=self.device)) 
        loss_n = self.classification_criterion(yn, torch.zeros_like(yn, device=self.device))
        loss_center = self.intra_utter_criterion(zp) + self.intra_utter_criterion(zn) + self.intra_utter_criterion(z)
        loss = loss_p + loss_n + 0.1 * loss_center

        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss_center', loss_center, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss_clf', loss_p+loss_n, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss_p', loss_p, on_step=False, on_epoch=True, prog_bar=False)
        self.log('loss_n', loss_n, on_step=False, on_epoch=True, prog_bar=False)

        return {'loss':loss}
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log('epoch_loss' , loss, prog_bar=True)

        # if self.current_epoch > 10:
        #     for param in self.E.feature_extractor.parameters():
        #         param.requires_grad = True

