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

import sys
sys.path.append("/home/tzuf_google_com/dev/cabby")

from typing import Tuple, Sequence, Optional, Dict, Text, Any

from absl import logging
import numpy as np
import os
import pandas as pd
from torchtext import data
import torch
import transformers
from transformers import DistilBertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

from shapely.geometry.point import Point

from cabby.geo import util as gutil
from cabby.model.text import util 

from cabby.geo import regions


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

CELLID_DIM = 64

class CabbyDataset(torch.utils.data.Dataset):
  def __init__(self, data: pd.DataFrame, s2level: int,
         cellid_to_label: Dict[int, int] = None):

    # Tokenize instructions.
    self.encodings = tokenizer(
      data.text.tolist(), truncation=True,
      padding=True, add_special_tokens=True)

    data['point'] = data.ref_point.apply(
      lambda x: gutil.point_from_str_coord(x))
    self.points = data.ref_point.apply(
      lambda x: gutil.coords_from_str_coord(x)).tolist()
    data['cellid'] = data.point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))

    data['neighbor_cells'] =data.cellid.apply(lambda x: gutil.neighbor_cellid(x))
    
    data['far_cells'] =  data.cellid.apply(lambda x: gutil.far_cellid(x))

    cellids_array = np.array(data.cellid.tolist())
    neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    far_cells_array = np.array(data.far_cells.tolist())

    self.cellids = util.binary_representation(cellids_array, dim = CELLID_DIM)
    self.neighbor_cells =  util.binary_representation(
      neighbor_cells_array, dim = CELLID_DIM)

    self.far_cells =  util.binary_representation(
      far_cells_array, dim = CELLID_DIM)

    
  def __getitem__(self, idx: int):
    text = {key: torch.tensor(val[idx])
        for key, val in self.encodings.items()}
    cellids = torch.tensor(self.cellids[idx])
    neighbor_cells = torch.tensor(self.neighbor_cells[idx])
    far_cells = torch.tensor(self.far_cells[idx])
    points = torch.tensor(self.points[idx])
    return text, cellids, neighbor_cells, far_cells, points

  def __len__(self):
    return len(self.cellids)


def create_dataset(data_dir: Text,
           region: Text, s2level) -> Tuple[CabbyDataset, CabbyDataset,
                           CabbyDataset, Dict[int, int]]:
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
    region: The region of the data.
    s2level: The s2level of the cells.
  Returns:
    The train, validate and test sets and the dictionary of labels to cellids.
  '''

  train_ds = pd.read_json(os.path.join(data_dir,'train.json'))
  valid_ds = pd.read_json(os.path.join(data_dir,'dev.json'))
  test_ds = pd.read_json(os.path.join(data_dir,'test.json'))

  # Get lables.
  get_region = regions.get_region(region)
  unique_cellid = gutil.cellids_from_polygon(get_region, s2level)
  label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellid)}
  cellid_to_label = {cellid: idx for idx, cellid in enumerate(unique_cellid)}

  # Create Cabby dataset.
  train_dataset = CabbyDataset(train_ds, s2level, cellid_to_label)
  val_dataset = CabbyDataset(valid_ds, s2level, cellid_to_label)
  test_dataset = CabbyDataset(test_ds, s2level, cellid_to_label)

  return train_dataset, val_dataset, test_dataset, label_to_cellid

# region = "Pittsburgh"
# dataset_dir = "~/data/wikigeo/pittsburgh"
# s2_level =12

# train_dataset, valid_dataset, test_dataset, label_to_cellid = create_dataset(dataset_dir, region, s2_level)
