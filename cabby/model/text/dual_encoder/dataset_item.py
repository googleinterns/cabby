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
from typing import Any, Dict, Text 
import torch
from transformers import DistilBertTokenizerFast

import attr

from cabby.model import util as mutil
from cabby.geo import util as gutil
from cabby.model.text import util 

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

CELLID_DIM = 64


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
    valid_path_dataset: Text, test_path_dataset: Text, 
    unique_cellid_path: Text, tensor_cellid_path: Text, 
    label_to_cellid_path: Text):

    logging.info("Loading dataset from <== {}.".format(dataset_path))
    train_dataset = torch.load(train_path_dataset)
    valid_dataset = torch.load(valid_path_dataset)
    test_dataset = torch.load(test_path_dataset)

    unique_cellid = np.load(unique_cellid_path, allow_pickle='TRUE')
    label_to_cellid = np.load(
      label_to_cellid_path, allow_pickle='TRUE').item()
    tens_cells = torch.load(tensor_cellid_path)
    n_cells = len(unique_cellid)
    dataset_text = TextGeoDataset(
      train_dataset, valid_dataset, test_dataset, 
      unique_cellid, tens_cells, label_to_cellid)

    return dataset_text
  
  @classmethod
  def save(cls, dataset_text: Any, dataset_path: Text,     
    train_path_dataset: Text, valid_path_dataset: Text,
    test_path_dataset: Text,  unique_cellid_path: Text, 
    tensor_cellid_path: Text, label_to_cellid_path: Text):

    os.mkdir(dataset_path)
    torch.save(dataset_text.train, train_path_dataset)
    torch.save(dataset_text.valid, valid_path_dataset)
    torch.save(dataset_text.test, test_path_dataset)
    np.save(unique_cellid_path, dataset_text.unique_cellids) 
    torch.save(dataset_text.unique_cellids_binary, tensor_cellid_path)
    np.save(label_to_cellid_path, dataset_text.label_to_cellid) 

    logging.info("Saved data to ==> {}.".format(dataset_path))


class TextGeoSplit(torch.utils.data.Dataset):
  """A split of of the RUN dataset.
  
  `points`: The ground true end-points of the samples.
  `labels`: The ground true label of the cellid.
  `cellids`: The ground truth S2Cell id.
  `neighbor_cells`: One neighbor cell id of the ground truth S2Cell id.
  `far_cells`: One far away cell id (in the region defined) of the ground truth 
  'dprob': Gamma distribution probability.
  S2Cell id.
  """
  def __init__(self, data: pd.DataFrame, s2level: int, 
    unique_cells_df: pd.DataFrame, cellid_to_label: Dict[int, int], 
    dprob: mutil.DistanceProbability):
    # Tokenize instructions.
    self.encodings = tokenizer(
      data.instructions.tolist(), truncation=True,
      padding=True, add_special_tokens=True)

    start_points = data.start_point.apply(
      lambda x: gutil.point_from_list_coord(x))
    
    dist_lists = start_points.apply(
      lambda start: unique_cells_df.point.apply(
        lambda end: gutil.get_distance_between_points(start, end))
        ).values.tolist()
    

    self.prob = []
    for dist_list in dist_lists:
      prob = []
      for dist in dist_list:
        prob.append(dprob(dist))
      self.prob.append(prob)

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
    prob = torch.tensor(self.prob[idx])
    
    sample = {'text': text, 'cellid': cellid, 'neighbor_cells': neighbor_cells, 
      'far_cells': far_cells, 'point': point, 'label': label, 'prob': prob}

    return sample

  def __len__(self):
    return len(self.cellids)

