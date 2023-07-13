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

import logging

import lightning as L
import torch
import torch.nn.functional as F
from lightning.fabric.loggers import TensorBoardLogger
from rich.progress import Progress

torchlogging = logging.getLogger("torch")
torchlogging.propagate = False
torchlogging.setLevel(logging.ERROR)


def regularization(model, loss, l1_strength=0.5, l2_strength=0.5):
    output_layer = -1
    if l1_strength > 0:
        l1_reg = model.sequential[output_layer].weight.abs().sum()
        loss += l1_strength * l1_reg
    if l2_strength > 0:
        l2_reg = model.sequential[output_layer].weight.pow(2).sum()
        loss += l2_strength * l2_reg
    return loss


class QuantFabricTrainer:
    def __init__(
        self,
        accelerator="cpu",
        devices="auto",
        strategy="auto",
        num_nodes=1,
        max_epochs=20,
        precision="32-true",
        dtype="float32",
        matmul_precision="medium",
    ) -> None:
        """A custom, minimal Lightning Fabric Trainer"""

        if "32" in dtype and torch.cuda.is_available():
            torch.set_float32_matmul_precision(matmul_precision)

        self.fabric = L.Fabric(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            num_nodes=num_nodes,
            precision=precision,
            loggers=TensorBoardLogger(root_dir="logs"),
        )
        self.fabric.launch()

        self._dtype = getattr(torch, dtype)
        self.max_epochs = max_epochs
        self.loss = None
        self.dataset = None
        self.model = None

    def fit(
        self,
        model,
        dataset,
        l1_strength: float = 0.1,
        l2_strength: float = 0.1,
    ) -> None:
        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset)

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.dataloader = self.fabric.setup_dataloaders(self.dataloader)

        self.model.train()
        with Progress() as progress:
            task = progress.add_task("training", total=self.max_epochs)
            while not progress.finished:
                for epoch in range(self.max_epochs):
                    for batch in self.dataloader:
                        input, target = batch
                        input = input.to(self._dtype)
                        self.optimizer.zero_grad()
                        output = self.model(input)
                        criterion = F.cross_entropy(output, target.to(torch.long))
                        self.loss = regularization(
                            self.model,
                            criterion,
                            l1_strength=l1_strength,
                            l2_strength=l2_strength,
                        )
                        self.fabric.log("loss", self.loss)
                        self.fabric.backward(self.loss)
                        self.optimizer.step()
                    progress.advance(task)
