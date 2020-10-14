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
'''Basic classes and functions for Wikigeo items.'''

from absl import logging
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import numpy as np
import os
import pandas as pd
import re
from shapely.geometry.point import Point
from shapely.geometry import box, mapping, LineString
import sys
from typing import Text, Dict, Any
import torch

import attr

@attr.s
class TextGeoDataset:
  """Construct a RVSPath sample.
  `train` is the train split.
  `valid` is the valid split.
  `test` is the test split.
  `unique_cellids` is the unique S2Cells.
  `unique_cellids_binary`  is the binary tensor of the unique S2Cells.
  `label_to_cellid` is the dictionary mapping labels to cellids.
  """
  train: Any = attr.ib()
  valid: Any = attr.ib()
  test: Any = attr.ib()
  unique_cellids: np.ndarray = attr.ib()
  unique_cellids_binary: torch.tensor = attr.ib()
  label_to_cellid: Dict[int, int] = attr.ib()


  @classmethod
  def from_TextGeoSplit(cls, train, valid, test, unique_cellids,
                              unique_cellids_binary, label_to_cellid):
    """Construct a TextGeoDataset."""
    return TextGeoDataset(
      train,
      valid,
      test,
      unique_cellids,
      unique_cellids_binary,
      label_to_cellid,
    )

  @classmethod
  def load(cls, dataset_path: Text, train_path_dataset: Text,
    valid_path_dataset: Text, unique_cellid_path: Text, 
    tensor_cellid_path: Text, label_to_cellid_path: Text):
    

    logging.info("Loading dataset from <== {}.".format(dataset_path))
    train_dataset = torch.load(train_path_dataset)
    valid_dataset = torch.load(valid_path_dataset)
    unique_cellid = np.load(unique_cellid_path, allow_pickle='TRUE')
    label_to_cellid = np.load(
      label_to_cellid_path, allow_pickle='TRUE').item()
    tens_cells = torch.load(tensor_cellid_path)
    n_cells = len(unique_cellid)
    dataset_text = TextGeoDataset(train_dataset, valid_dataset, None, unique_cellid, tens_cells, label_to_cellid)

    return dataset_text
  
  @classmethod
  def save(cls, dataset_text: Any, dataset_path: Text,     
    train_path_dataset: Text, valid_path_dataset: Text, 
    unique_cellid_path: Text, tensor_cellid_path: Text,
    label_to_cellid_path: Text):

    os.mkdir(dataset_path)
    torch.save(dataset_text.train, train_path_dataset)
    torch.save(dataset_text.valid, valid_path_dataset)
    np.save(unique_cellid_path, dataset_text.unique_cellid) 
    torch.save(dataset_text.tens_cells, tensor_cellid_path)
    np.save(label_to_cellid_path, dataset_text.label_to_cellid) 

    logging.info("Saved data to ==> {}.".format(dataset_path))
