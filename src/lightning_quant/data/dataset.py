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

import pandas as pd
import torch
from torch.utils.data import Dataset


class MarketDataset(Dataset):
    def __init__(self, featuresdir: str = "data/features", labelsdir: str = "data/labels", labelcol: str = "position"):
        featuresdir: str = "data/features"
        labelsdir: str = "data/labels"
        features = pd.read_parquet(featuresdir).reset_index()
        labels = pd.read_parquet(labelsdir).reset_index()
        self.data = labels.merge(features, on="index")
        self.data.drop(["symbol", "vwap"], axis=1, inplace=True)
        self.data.dropna(inplace=True)
        self.data.set_index("index", inplace=True)
        self.labelcol = labelcol
        self.featurecols = [i for i in self.data.columns if i != self.labelcol]
        self.labelweights = self.data[self.labelcol].value_counts()
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[self.featurecols].iloc[idx], self.data[self.labelcol].iloc[idx]
        return torch.tensor(x, dtype=torch.float16), torch.tensor(y, dtype=torch.float16)
