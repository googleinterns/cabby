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
from sklearn.utils import shuffle
import torch

from cabby.geo import util as gutil
from cabby.model.text import util 

from cabby.geo import regions
from cabby.model.text.dual_encoder import dataset_item
from cabby.model import datasets
from cabby.model import util as mutil

# DISTRIBUTION_SCALE_DISTANCEA is a factor (in meters) that gives the overall 
# scale in meters for the distribution.
DISTRIBUTION_SCALE_DISTANCE = 37
dprob = mutil.DistanceProbability(DISTRIBUTION_SCALE_DISTANCE)

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
  run_dataset = datasets.RUNDataset(data_dir, s2level)

  points = gutil.get_centers_from_s2cellids(run_dataset.unique_cellid)

  unique_cells_df = pd.DataFrame(
    {'point': points, 'cellid': run_dataset.unique_cellid})
  
  unique_cells_df['far'] = unique_cells_df.point.apply(
      lambda x: gutil.far_cellid(x, unique_cells_df))

  vec_cells = util.binary_representation(unique_cells_df.cellid.to_numpy(), 
  dim = dataset_item.CELLID_DIM)
  tens_cells = torch.tensor(vec_cells)

  # Create RUN dataset.
  train_dataset = None
  val_dataset = None
  logging.info("Starting to create the splits")
  if infer_only == False:
    train_dataset = dataset_item.TextGeoSplit(
      run_dataset.train, s2level, unique_cells_df, 
      run_dataset.cellid_to_label, dprob)
    logging.info(
      f"Finished to create the train-set with {len(train_dataset)} samples")
    val_dataset = dataset_item.TextGeoSplit(
      run_dataset.valid, s2level, unique_cells_df, 
      run_dataset.cellid_to_label, dprob)
    logging.info(
      f"Finished to create the valid-set with {len(val_dataset)} samples")
  test_dataset = dataset_item.TextGeoSplit(
    run_dataset.test, s2level, unique_cells_df, 
    run_dataset.cellid_to_label, dprob)
  logging.info(
    f"Finished to create the test-set with {len(test_dataset)} samples")

  return dataset_item.TextGeoDataset.from_TextGeoSplit(
    train_dataset, val_dataset, test_dataset, 
    np.array(run_dataset.unique_cellid), 
    tens_cells, run_dataset.label_to_cellid)

