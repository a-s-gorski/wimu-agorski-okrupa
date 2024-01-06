import lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy

from src.models.few_shot_learner import FewShotLearner


class FewShotLearnerKSoft(FewShotLearner):

    def step(self, batch, batch_idx, tag: str):
        support, unlabeled, query = batch

        logits = self.protonet(support, unlabeled, query)
        loss = self.loss(logits, query["target"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output
