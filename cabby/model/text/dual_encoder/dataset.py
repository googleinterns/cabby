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
import random
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
from cabby.model.text.dual_encoder import dataset_item


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

CELLID_DIM = 64



class TextGeoSplit(torch.utils.data.Dataset):
  def __init__(self, data: pd.DataFrame, s2level: int, 
    cells: int, cellid_to_label: Dict[int, int]):
    
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


    data['neighbor_cells'] = data.cellid.apply(
      lambda x: gutil.get_cell_neighbors(x))
    
    self.neighbor_cells = data['neighbor_cells']

    self.far_cells = data.point.apply(lambda x: gutil.far_cellids(x, cells))
    

    cellids_array = np.array(data.cellid.tolist())
    # neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    # far_cells_array = np.array(data.far_cells.tolist())

    self.cellids = util.binary_representation(cellids_array, dim = CELLID_DIM)
    # self.neighbor_cells =  util.binary_representation(
      # neighbor_cells_array, dim = CELLID_DIM)

    # self.far_cells =  util.binary_representation(
    #   far_cells_array, dim = CELLID_DIM)

    
  def __getitem__(self, idx: int):
    '''Supports indexing such that TextGeoDataset[i] can be used to get 
    i-th sample. 
    Arguments:
      idx: The index for which a sample from the dataset will be returned.
    Returns:
      A single sample including text, the correct cellid, a neighbor cellid, 
      a far cellid, a point of the cellid and the label of the cellid.
    '''
    text = {key: torch.tensor(val[idx])
        for key, val in self.encodings.items()}
    cellid = torch.tensor(self.cellids[idx])
    cell_neighbor = random.choice(self.neighbor_cells[idx])
    neighbor_cell_array = np.array([cell_neighbor])
    neighbor_cells_binary =  util.binary_representation(
      neighbor_cell_array, dim = CELLID_DIM)
    neighbor_cells = torch.tensor(neighbor_cells_binary).squeeze(0)


    cell_far = random.choice(self.far_cells[idx])
    far_cell_array = np.array([cell_far])
    far_cells_binary =  util.binary_representation(
      far_cell_array, dim = CELLID_DIM)
    far_cells = torch.tensor(far_cells_binary).squeeze(0)

    point = torch.tensor(self.points[idx])
    label = torch.tensor(self.labels[idx])
    
    # print(far_cells.shape,neighbor_cells.shape)
    sample = {'text': text, 'cellid': cellid, 'neighbor_cells': neighbor_cells, 
      'far_cells': far_cells, 'point': point, 'label': label}

    return sample

  def __len__(self):
    return len(self.cellids)


def create_dataset(data_dir: Text, region: Text, s2level: int
) -> dataset_item.TextGeoDataset:
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
    region: The region of the data.
    s2level: The s2level of the cells.
  Returns:
    The train, validate and test sets and the dictionary of labels to cellids.
  '''

  train_ds = pd.read_json(os.path.join(data_dir, 'train.json'))
  valid_ds = pd.read_json(os.path.join(data_dir, 'dev.json'))
  test_ds = pd.read_json(os.path.join(data_dir, 'test.json'))

  # Get labels.
  get_region = regions.get_region(region)
  unique_cellid = gutil.cellids_from_polygon(get_region, s2level)
  label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellid)}
  cellid_to_label = {cellid: idx for idx, cellid in enumerate(unique_cellid)}

  points = gutil.get_centers_from_s2cellids(unique_cellid)
  cells = pd.DataFrame({'point': points, 'cellid': unique_cellid})
  vec_cells = util.binary_representation(cells.cellid.to_numpy(), 
  dim = CELLID_DIM)
  tens_cells = torch.tensor(vec_cells)

  # Create Cabby dataset.
  train_dataset = TextGeoSplit(train_ds, s2level, cells, cellid_to_label)
  val_dataset = TextGeoSplit(valid_ds, s2level, cells, cellid_to_label)
  test_dataset = TextGeoSplit(test_ds, s2level, cells, cellid_to_label)

  return dataset_item.TextGeoDataset.from_TextGeoSplit(
    train_dataset, val_dataset, test_dataset, np.array(unique_cellid), 
    tens_cells, label_to_cellid)

