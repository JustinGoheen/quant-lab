# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lightning as L
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy


class LogisticRegression(L.LightningModule):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bias: bool = True,
        lr: float = 0.001,
        optimizer="Adam",
        accuracy_task: str = "multiclass",
    ):
        self.model = nn.Linear(in_features=in_features, out_features=num_classes, bias=bias)
        self.loss = accuracy
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.accuracy_task = accuracy_task
        self.num_classes = num_classes
        self.save_hyperparameters()

        super().__init__()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        self.common_step(batch, "training")

    def test_step(self, batch, *args):
        self.common_step(batch, "test")

    def validation_step(self, batch, *args):
        self.common_step(batch, "val")

    def common_step(self, batch, stage):
        """consolidates common code for train, test, and validation steps"""
        x, y = batch
        y_hat = self.model(x)
        criterion = F.cross_entropy(y_hat, y)
        loss = self._regularization(criterion)

        if stage == "training":
            self.log(f"{stage}_loss", loss)
            return loss
        if stage in ["val", "test"]:
            acc = accuracy(y_hat.argmax(dim=-1), y, task=self.accuracy_task, num_classes=self.num_classes)
            self.log(f"{stage}_acc", acc)
            self.log(f"{stage}_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)

    def configure_optimizers(self):
        """configures the ``torch.optim`` used in training loop"""
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def _regularization(self, loss):
        """borrowed from lightning bolts"""
        if self.hparams.l1_strength > 0:
            l1_reg = self.model.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        if self.hparams.l2_strength > 0:
            l2_reg = self.model.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg
        return loss
