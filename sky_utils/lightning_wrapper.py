# pytorch
import torch
import torch.nn.functional as F

# pytorch lightning
import pytorch_lightning as pl
import wandb
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1_score, fbeta_score

# custom models
from sky_utils.models import *


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
        preds = torch.where(logits > 0.5, 1, 0)
        acc = accuracy(preds, y)
        p = precision(preds, y)
        r = recall(preds, y)
        f1 = f1_score(preds, y)
        self.log('train_f1_score', f1, on_epoch=True, on_step=False, logger=True)
        self.log('train_precision', p, on_epoch=True, on_step=False, logger=True)
        self.log('train_recall', r, on_epoch=True, on_step=False, logger=True)
        self.log('train_loss', loss, on_epoch=True, on_step=False, logger=True)
        self.log('train_acc', acc, on_epoch=True, on_step=False, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # validation metrics
        preds = torch.where(logits > 0.5, 1, 0)
        acc = accuracy(preds, y)
        p = precision(preds, y)
        r = recall(preds, y)
        f1 = f1_score(preds, y)
        self.log('val_f1_score', f1, on_epoch=True, prog_bar=True)
        self.log('val_precision', p, on_epoch=True, prog_bar=True)
        self.log('val_recall', r, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

        return preds, y

    # logic for a single test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # test metrics
        preds = torch.where(logits > 0.5, 1, 0)
        acc = accuracy(preds, y)
        p = precision(preds, y)
        r = recall(preds, y)
        f1 = f1_score(preds, y)
        self.log('test_f1_score', f1, on_epoch=True, prog_bar=True)
        self.log('test_precision', p, on_epoch=True, prog_bar=True)
        self.log('test_recall', r, on_epoch=True, prog_bar=True)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        f1_p_r_heatmap = torch.zeros((3, 38))

        preds_stacked = validation_step_outputs[0][0]
        y_stacked = validation_step_outputs[0][1]

        for i in range(1, len(validation_step_outputs)):
            preds, y = validation_step_outputs[i]
            preds_stacked = torch.cat((preds_stacked, preds))
            y_stacked = torch.cat((y_stacked, y))

        # F1
        for i in range(f1_p_r_heatmap.size()[1]):
            f1_p_r_heatmap[0][i] = f1_score(preds_stacked[:, i], y_stacked[:, i])

        # Precision
        for i in range(f1_p_r_heatmap.size()[1]):
            f1_p_r_heatmap[1][i] = precision(preds_stacked[:, i], y_stacked[:, i])

        # Recall
        for i in range(f1_p_r_heatmap.size()[1]):
            f1_p_r_heatmap[2][i] = recall(preds_stacked[:, i], y_stacked[:, i])

        class_names = [
            "Amusement park", "Animals", "Bench", "Building", "Castle", "Cave", "Church", "City", "Cross",
            "Cultural institution", "Food", "Footpath", "Forest", "Furniture", "Grass", "Graveyard", "Lake",
            "Landscape", "Mine", "Monument", "Motor vehicle", "Mountains", "Museum", "Open-air museum", "Park",
            "Person", "Plants", "Reservoir", "River", "Road", "Rocks", "Snow", "Sport", "Sports facility", "Stairs",
            "Trees", "Watercraft", "Windows"
        ]

        self.logger.experiment.log({
            "f1_p_r_heatmap": wandb.plots.HeatMap(
                class_names,
                ["f1", "precision", "recall"],
                f1_p_r_heatmap.tolist(),
                show_text=True
            )
        })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0.00005)
