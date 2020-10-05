# Copyright 2020 The Flax Authors.
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

from typing import Tuple, Sequence, Optional, Dict, Text, Any

from absl import logging
import numpy as np
from numpy import int64
import os
import pandas as pd
from torchtext import data
import torch
import transformers
from transformers import DistilBertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


import sys
sys.path.append("/home/tzuf_google_com/dev/cabby")

from cabby.geo import util
from cabby.geo.map_processing import map_structure


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


class CabbyDataset(torch.utils.data.Dataset):
  def __init__(self, data: pd.DataFrame, s2level: int,
         cellid_to_label: Dict[int64, int64] = None):

    # Tokenize instructions.
    self.encodings = tokenizer(
      data.instruction.tolist(), truncation=True,
      padding=True, add_special_tokens=True)

    data['point'] = data.end_point.apply(
      lambda x: util.point_from_list_yx(x))
    self.coords = data.end_point.tolist()
    data['cellid'] = data.point.apply(
      lambda x: util.cellid_from_point(x, s2level))

    self.labels = data.cellid.apply(lambda x: cellid_to_label[x]).tolist()

  def __getitem__(self, idx: int):
    item = {key: torch.tensor(val[idx])
        for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    coords = torch.tensor(self.coords[idx])
    return item, coords

  def __len__(self):
    return len(self.labels)


def create_dataset(path: Text, cellid_to_label_path: Text, lables_dictionary_path: Text, s2level: int) -> Tuple[CabbyDataset, Dict[int64,int64]]:
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
    region: The region of the data.
    s2level: The s2level of the cells.
  Returns:
    The train, validate and test sets and the dictionary of labels to cellids.
  '''

  valid_ds = pd.read_json(path)
  cellid_to_label = np.load(cellid_to_label_path, 
    allow_pickle='TRUE').item() 

  lables_dictionary = np.load(lables_dictionary_path, 
    allow_pickle='TRUE').item() 

  # Create Cabby dataset.
  valid_ds = CabbyDataset(valid_ds, s2level, cellid_to_label)

  return valid_ds, lables_dictionary

