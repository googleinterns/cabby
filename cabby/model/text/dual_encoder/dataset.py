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
import os
import pandas as pd
import sys
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
  def __init__(self, data: pd.DataFrame, s2level: int, cells: int, cellid_to_label: Dict[int, int]):
    
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
    self.labels = data.cellid.apply(lambda x: cellid_to_label[x]).tolist()


    data['neighbor_cells'] =data.cellid.apply(lambda x: gutil.neighbor_cellid(x))
    
    far_cellids =  data.point.apply(lambda x: gutil.far_cellid(x, cells))
    if far_cellids is None:
      sys.exit("Far cellid was not found.")
    data['far_cells'] = far_cellids


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
    labels = torch.tensor(self.labels[idx])

    return text, cellids, neighbor_cells, far_cells, points, labels

  def __len__(self):
    return len(self.cellids)


def create_dataset(data_dir: Text,
           region: Text, s2level: int) -> Tuple[CabbyDataset, CabbyDataset,
                           CabbyDataset, np.ndarray, Any]:
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

  points = gutil.get_cell_center_point_from_s2cellids(unique_cellid)
  cells = pd.DataFrame({'point': points, 'cellid': unique_cellid})
  vec_cellls = util.binary_representation(cells.cellid.to_numpy(), dim = CELLID_DIM)
  tens_cells = torch.tensor(vec_cellls)

  # Create Cabby dataset.
  train_dataset = CabbyDataset(train_ds, s2level, cells, cellid_to_label)
  val_dataset = CabbyDataset(valid_ds, s2level, cells, cellid_to_label)
  test_dataset = CabbyDataset(test_ds, s2level, cells, cellid_to_label)

  return train_dataset, val_dataset, test_dataset, np.array(unique_cellid), tens_cells
