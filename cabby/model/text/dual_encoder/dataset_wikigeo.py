# coding=utf-8
# Copyright 2020 Google LLC
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

from typing import Dict

from absl import logging
import numpy as np
import os
import pandas as pd

import torch
from transformers import DistilBertTokenizerFast

from cabby.geo import util as gutil
from cabby.model.text import util 

from cabby.geo import regions
from cabby.model.text.dual_encoder import dataset_item


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

CELLID_DIM = 64

class TextGeoSplit(torch.utils.data.Dataset):
  def __init__(self, data: pd.DataFrame, s2level: int, 
    unique_cells_df: pd.DataFrame, cellid_to_label: Dict[int, int]):
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
      lambda x: gutil.neighbor_cellid(x))

    data['far_cells'] = data.cellid.apply(
      lambda cellid: unique_cells_df[unique_cells_df['cellid']==cellid].far.iloc[0])

    cellids_array = np.array(data.cellid.tolist())
    neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    far_cells_array = np.array(data.far_cells.tolist())

    self.cellids = util.binary_representation(cellids_array, dim = CELLID_DIM)
    self.neighbor_cells =  util.binary_representation(
      neighbor_cells_array, dim = CELLID_DIM)

    self.far_cells =  util.binary_representation(
      far_cells_array, dim = CELLID_DIM)

    
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
    neighbor_cells = torch.tensor(self.neighbor_cells[idx])
    far_cells = torch.tensor(self.far_cells[idx])
    point = torch.tensor(self.points[idx])
    label = torch.tensor(self.labels[idx])
    
    sample = {'text': text, 'cellid': cellid, 'neighbor_cells': neighbor_cells, 
      'far_cells': far_cells, 'point': point, 'label': label}

    return sample

  def __len__(self):
    return len(self.cellids)

def create_dataset(
          data_dir: str, 
          region: str, 
          s2level: int, 
          infer_only: bool = False
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
  unique_cellid = gutil.cellids_from_polygon(get_region.polygon, s2level)
  label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellid)}
  cellid_to_label = {cellid: idx for idx, cellid in enumerate(unique_cellid)}

  points = gutil.get_centers_from_s2cellids(unique_cellid)

  unique_cells_df = pd.DataFrame({'point': points, 'cellid': unique_cellid})
  
  unique_cells_df['far'] = unique_cells_df.point.apply(
      lambda x: gutil.far_cellid(x, unique_cells_df))

  vec_cells = util.binary_representation(unique_cells_df.cellid.to_numpy(), 
  dim = CELLID_DIM)
  tens_cells = torch.tensor(vec_cells)

  # Create WikiGeo dataset.
  train_dataset = None
  val_dataset = None
  logging.info("Starting to create the splits")
  if infer_only == False:
    train_dataset = TextGeoSplit(
      train_ds, s2level, unique_cells_df, cellid_to_label)
    logging.info(
      f"Finished to create the train-set with {len(train_dataset)} samples")
    val_dataset = TextGeoSplit(
      valid_ds, s2level, unique_cells_df, cellid_to_label)
    logging.info(
      f"Finished to create the valid-set with {len(val_dataset)} samples")
  test_dataset = TextGeoSplit(
    test_ds, s2level, unique_cells_df, cellid_to_label)
  logging.info(
    f"Finished to create the test-set with {len(test_dataset)} samples")

  return dataset_item.TextGeoDataset.from_TextGeoSplit(
    train_dataset, val_dataset, test_dataset, np.array(unique_cellid), 
    tens_cells, label_to_cellid)

