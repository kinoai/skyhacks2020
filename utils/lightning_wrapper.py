# pytorch
import torch
import torch.nn.functional as F

# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score

# custom models
from utils.models import *


class LitModel(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters(config["hparams"])
        # self.model = MNISTExampleModel(config=self.hparams)
        # self.model = ResNetTrasferLearning(config=self.hparams)
        self.model = EfficientNetTransferLearning(config=self.hparams)

        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # training metrics
        preds = torch.where(logits > 0.5, torch.FloatTensor([1]), torch.FloatTensor([0]))
        acc = accuracy(preds, y)
        p = precision(preds, y)
        r = recall(preds, y)
        f1 = f1_score(preds, y)
        self.log('train_f1_score', f1, on_epoch=True, on_step=False)
        self.log('train_precision', p, on_epoch=True, on_step=False)
        self.log('train_recall', r, on_epoch=True, on_step=False)
        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_acc', acc, on_epoch=True, on_step=False)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # training metrics
        preds = torch.where(logits > 0.5, torch.FloatTensor([1]), torch.FloatTensor([0]))
        acc = accuracy(preds, y)
        p = precision(preds, y)
        r = recall(preds, y)
        f1 = f1_score(preds, y)
        self.log('val_f1_score', f1, on_epoch=True, prog_bar=True)
        self.log('val_precision', p, on_epoch=True, prog_bar=True)
        self.log('val_recall', r, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)