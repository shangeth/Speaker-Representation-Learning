import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from models.encoder import Encoder2, Discriminator, Accumulator
from pytorch_lightning.metrics.classification import Accuracy
from wavencoder.models import Wav2Vec

# from encoder import Encoder2
class SpeakerRecognitionModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()

        self.encoder = Encoder(HPARAMS['hidden_dim'])
        self.encoder = Wav2Vec(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Linear(HPARAMS['hidden_dim'], HPARAMS['out_dim']),
            nn.LogSoftmax(1)
            )

        self.classification_criterion = nn.NLLLoss()
        self.accuracy = Accuracy()
        self.lr=1e-3
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        attn_output = x.transpose(1, 2)

        speaker = self.classifier(attn_output)
        return speaker
    
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-3, 
        # cycle_momentum=False,step_size_down=2000, step_size_up=2000)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id
        x, y = batch
        y_hat = self(x)
        y = y.view(-1).float()
        y_hat = y_hat.view(-1).float()

        loss = self.classification_criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return {'loss':loss, 
                'train_acc':acc,
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        acc = torch.tensor([x['train_acc'] for x in outputs]).mean()
        self.log('epoch_loss' , loss, prog_bar=True)
        self.log('acc',acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = y.view(-1).float()
        y_hat = y_hat.view(-1).float()

        loss = self.classification_criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())

        return {'val_loss':loss, 
                'val_acc':acc}

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        acc = torch.tensor([x['val_acc'] for x in outputs]).mean()
        
        self.log('v_loss' , val_loss, prog_bar=True)
        self.log('v_acc',acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x,  y = batch
        y_hat = self(x)
        y = y.view(-1).float()
        y_hat = y_hat.view(-1).float()

        loss = self.classification_criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds.long(), y.long())
        
        return {'test_acc':acc}

    def test_epoch_end(self, outputs):
        acc = torch.tensor([x['test_acc'] for x in outputs]).mean()
        pbar = {'test_acc':acc.item()}
        # self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
        # return {'log': pbar, 'progress_bar':pbar}

class RepresentationModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        self.save_hyperparameters()

        # self.E = nn.Sequential(
        #     Wav2Vec(pretrained=True),
        #     nn.BatchNorm1d(512),
        #     nn.Conv1d(512, 128, 1),
        #     nn.BatchNorm1d(128),)

        # self.E = nn.Sequential(
        #     nn.Conv1d(40, 128, 5),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 256, 5),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     # nn.Conv1d(512, 512, 3),
        #     # nn.BatchNorm1d(512),
        #     # nn.ReLU(),
        #     # nn.Conv1d(512, 512, 3),
        #     # nn.BatchNorm1d(512),
        #     # nn.ReLU()
        #     )
        self.E = Encoder2(inchannel=40, channels=HPARAMS['hidden_dim'])

        # self.E = Encoder(HPARAMS['hidden_dim'])
        self.A = Accumulator(HPARAMS['hidden_dim']*3)
        self.D = Discriminator(HPARAMS['hidden_dim']*6)

        self.classification_criterion = nn.BCELoss()
        self.lr=1e-3
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.E(x)
        return x
    
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, 
        # cycle_momentum=False,step_size_down=2000, step_size_up=2000)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer]
        # , [scheduler]

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

        # print(z.shape)
        loss_p = self.classification_criterion(yp, torch.ones_like(yp, device=self.device)) 
        loss_n = self.classification_criterion(yn, torch.zeros_like(yn, device=self.device))
        # print(loss_p.item(), loss_n.item(), yp[0].item(), yn[0].item())
        loss = loss_p + loss_n
        # print('LOSS = ', loss.item())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return {'loss':loss}
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log('epoch_loss' , loss, prog_bar=True)

