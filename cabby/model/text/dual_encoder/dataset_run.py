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
  """A split of of the RUN dataset.
  
  `points`: The ground true end-points of the samples.
  `labels`: The ground true label of the cellid.
  `cellids`: The ground truth S2Cell id.
  `neighbor_cells`: One neighbor cell id of the ground truth S2Cell id.
  `far_cells`: One far away cell id (in the region defined) of the ground truth 
  S2Cell id.
  """
  def __init__(self, data: pd.DataFrame, s2level: int, 
    unique_cells_df: pd.DataFrame, cellid_to_label: Dict[int, int]):
    # Tokenize instructions.
    self.encodings = tokenizer(
      data.instructions.tolist(), truncation=True,
      padding=True, add_special_tokens=True)

    points = data.end_point.apply(
      lambda x: gutil.point_from_list_coord(x))

    data = data.assign(point=points)

    data['cellid'] = data.point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))

    data['neighbor_cells'] = data.cellid.apply(
      lambda x: gutil.neighbor_cellid(x))

    data['far_cells'] = data.cellid.apply(
      lambda cellid: unique_cells_df[unique_cells_df['cellid']==cellid].far.iloc[0])

    cellids_array = np.array(data.cellid.tolist())
    neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    far_cells_array = np.array(data.far_cells.tolist())


    self.points = data.point.apply(
      lambda x: gutil.tuple_from_point(x)).tolist()
    self.labels = data.cellid.apply(lambda x: cellid_to_label[x]).tolist()
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
  ds = pd.read_json(os.path.join(data_dir, 'dataset.json'))
  ds['instructions'] = ds.groupby(
    ['id'])['instruction'].transform(lambda x: ' '.join(x))

  ds = ds.drop_duplicates(subset='id', keep="last")

  columns_keep = ds.columns.difference(
    ['map', 'id', 'instructions', 'end_point', 'start_point'])
  ds.drop(columns_keep, 1, inplace=True)

  dataset_size = ds.shape[0]
  logging.info(f"Size of dataset: {ds.shape[0]}")

  train_ds = ds[ds['map']=='map_1']
  valid_ds = ds[ds['map']=='map_2']
  test_ds = ds[ds['map']=='map_3']

  # Get unique cells.
  cells = ds.end_point.apply(
    lambda x: gutil.cellid_from_point(gutil.point_from_list_coord(x), s2level)
    ).tolist()
  unique_cellid = list(set(cells))

  # Get unique cells for test split.
  active_region_test = regions.get_region("RUN-map3")
  unique_cellid_test = gutil.cellids_from_polygon(
    active_region_test.polygon, s2level)
  label_to_cellid_test = {
    idx: cellid for idx, cellid in enumerate(unique_cellid_test)}
  cellid_to_label_test = {
    cellid: idx for idx, cellid in enumerate(unique_cellid_test)}
  label_to_cellid_test = {
    idx: cellid for idx, cellid in enumerate(unique_cellid_test)}
  points_test = gutil.get_centers_from_s2cellids(unique_cellid_test)

  unique_cells_df_test = pd.DataFrame(
    {'point': points_test, 'cellid': unique_cellid_test})

  vec_cells_test = util.binary_representation(
    unique_cells_df_test.cellid.to_numpy(), dim = CELLID_DIM)
  tens_cells_test = torch.tensor(vec_cells_test)
  logging.info(
    f"Shape of unique cells tensors in test: {tens_cells_test.shape}")


  # Get unique cells for valid split.
  active_region_valid = regions.get_region("RUN-map2")
  unique_cellid_valid = gutil.cellids_from_polygon(
    active_region_valid.polygon, s2level)
  label_to_cellid_valid = {
    idx: cellid for idx, cellid in enumerate(unique_cellid_valid)}
  cellid_to_label_valid = {
    cellid: idx for idx, cellid in enumerate(unique_cellid_valid)}
  label_to_cellid_valid = {
    idx: cellid for idx, cellid in enumerate(unique_cellid_valid)}
  points_valid = gutil.get_centers_from_s2cellids(unique_cellid_valid)

  unique_cells_df_valid = pd.DataFrame(
    {'point': points_valid, 'cellid': unique_cellid_valid})

  vec_cells_valid = util.binary_representation(
    unique_cells_df_valid.cellid.to_numpy(), dim = CELLID_DIM)
  tens_cells_valid = torch.tensor(vec_cells_valid)
  logging.info(
    f"Shape of unique cells tensors in test: {tens_cells_valid.shape}")



  # Get unique cells for train split.
  active_region_train = regions.get_region("RUN-map1")
  unique_cellid_train = gutil.cellids_from_polygon(
    active_region_train.polygon, s2level)
  label_to_cellid_train = {
    idx: cellid for idx, cellid in enumerate(unique_cellid_train)}
  cellid_to_label_train = {
    cellid: idx for idx, cellid in enumerate(unique_cellid_train)}
  label_to_cellid_train = {
    idx: cellid for idx, cellid in enumerate(unique_cellid_train)}
  points_train = gutil.get_centers_from_s2cellids(unique_cellid_train)

  unique_cells_df_train = pd.DataFrame(
    {'point': points_train, 'cellid': unique_cellid_train})

  vec_cells_train = util.binary_representation(
    unique_cells_df_train.cellid.to_numpy(), dim = CELLID_DIM)
  tens_cells_train = torch.tensor(vec_cells_train)
  logging.info(
    f"Shape of unique cells tensors in test: {tens_cells_train.shape}")


  # Create RUN dataset.
  train_dataset = None
  val_dataset = None
  logging.info("Starting to create the splits")
  if infer_only == False:
    train_dataset = TextGeoSplit(
      train_ds, s2level, unique_cells_df_train, cellid_to_label_train)
    logging.info(
      f"Finished to create the train-set with {len(train_dataset)} samples")
    val_dataset = TextGeoSplit(
      valid_ds, s2level, unique_cells_df_valid, cellid_to_label_valid)
    logging.info(
      f"Finished to create the valid-set with {len(val_dataset)} samples")
  test_dataset = TextGeoSplit(
    test_ds, s2level, unique_cells_df_test, cellid_to_label_test)
  logging.info(
    f"Finished to create the test-set with {len(test_dataset)} samples")
  
  return dataset_item.TextGeoDataset.from_TextGeoSplit(
    train_dataset, val_dataset, test_dataset, np.array(unique_cellid_test), 
    tens_cells_test, label_to_cellid_test)

