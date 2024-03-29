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

import multiprocessing
import os

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler

from quantlab.data.dataset import MarketDataset

NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class MarketDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset = MarketDataset,
        datapath: str = "data/SPY_2023-07-05_22:43:46.227411-04:00",
        train_size: float = 0.8,
        num_workers: int = NUMWORKERS,
        labelcol: str = "position",
    ):
        super().__init__()
        self.dataset = dataset(datapath=datapath, labelcol=labelcol)
        self.datapath = datapath
        self.train_size = train_size
        self.num_workers = num_workers
        self.featurespath = os.path.join(os.getcwd(), datapath, "features.pq")
        self.labelspath = os.path.join(os.getcwd(), datapath, "labels.pq")
        self.labelcol = labelcol
        self.sampler = None

    def prepare_data(self):
        self.dataset = self.dataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_size = int(len(self.dataset) * self.train_size)
            val_size = len(self.dataset) - train_size
            self.train_data, self.val_data = random_split(self.dataset, lengths=[train_size, val_size])
            self.sampler = WeightedRandomSampler(self.dataset.labelweights, train_size)
        if stage == "test" or stage is None:
            self.test_data = self.val_data

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, shuffle=False, sampler=self.sampler)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers, shuffle=False, sampler=self.sampler)

    def test_dataloader(self):
        return DataLoader(self.dataset, num_workers=self.num_workers, shuffle=False)
