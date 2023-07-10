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
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchmetrics.functional import accuracy


class ElasticNet(L.LightningModule):
    """Logistic Regression with L1 and L2 Regularization"""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        bias: bool = False,
        lr: float = 0.001,
        l1_strength: float = 0.1,
        l2_strength: float = 0.1,
        optimizer="Adam",
        accuracy_task: str = "multiclass",
        dtype=torch.float32,
    ):
        super().__init__()
        self.model = nn.Linear(in_features=in_features, out_features=num_classes, bias=bias, dtype=dtype)
        self.optimizer = getattr(optim, optimizer)
        self.lr = lr
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.accuracy_task = accuracy_task
        self.num_classes = num_classes
        self._dtype = dtype  # cannot set explicitly
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.model(x.to(self._dtype))

    def training_step(self, batch):
        return self.common_step(batch, "training")

    def test_step(self, batch, *args):
        self.common_step(batch, "test")

    def validation_step(self, batch, *args):
        self.common_step(batch, "val")

    def common_step(self, batch, stage):
        """consolidates common code for train, test, and validation steps"""
        x, y = batch
        x = x.to(self._dtype)
        y = y.to(torch.long)  # cross_entropy expect long int64
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
        optimizer = self.optimizer(
            self.parameters(),
            lr=self.lr,
        )
        return optimizer

    def _regularization(self, loss):
        """borrowed from lightning bolts"""
        if self.hparams.l1_strength > 0:
            l1_reg = self.model.weight.abs().sum()
            loss += self.l1_strength * l1_reg

        if self.hparams.l2_strength > 0:
            l2_reg = self.model.weight.pow(2).sum()
            loss += self.l2_strength * l2_reg
        return loss