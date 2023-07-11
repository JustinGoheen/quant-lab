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

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from lightning.pytorch.profilers import Profiler


class QuantLightningTrainer(L.Trainer):
    """A custom Lightning.LightningTrainer

    # Arguments
        logger: None
        profiler: None
        callbacks: []
        plugins: []
        set_seed: True
        trainer_init_kwargs:
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        profiler: Optional[Profiler] = None,
        callbacks: Optional[List] = [],
        plugins: Optional[List] = [],
        set_seed: bool = True,
        seed: int = 42,
        profiler_logs: str = "logs/torch_profiler",
        tensorboard_logs: str = "logs/tensorboard",
        checkpoints_dir: str = "models/checkpoints",
        **trainer_init_kwargs: Dict[str, Any]
    ) -> None:
        if set_seed:
            seed_everything(seed, workers=True)

        super().__init__(
            logger=logger or TensorBoardLogger(tensorboard_logs, name="logs"),
            profiler=profiler,
            callbacks=callbacks + [ModelCheckpoint(dirpath=checkpoints_dir, filename="model")],
            plugins=plugins,
            **trainer_init_kwargs
        )

    def persist_predictions(self, preds_path: Optional[Union[str, Path]] = "models/preds.pt") -> None:
        """helper method to persist predictions on completion of a training run

        # Arguments
            preds_path: the file path where predictions should be saved to
        """
        self.test(ckpt_path="best", datamodule=self.datamodule)
        predictions = self.predict(self.model, self.datamodule.test_dataloader())
        torch.save(predictions, preds_path)
