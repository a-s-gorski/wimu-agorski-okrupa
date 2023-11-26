import lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy


class FewShotLearner(pl.LightningModule):

    def __init__(self, protonet: nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.protonet = protonet
        self.learning_rate = learning_rate

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy()
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits = self.protonet(support, query)
        loss = self.loss(logits, query["target"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")
