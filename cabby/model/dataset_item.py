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

from xmlrpc.client import Boolean
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
import swifter
from typing import Any, Dict, Text 
import torch

import attr

from cabby.geo import util as gutil
from cabby.model import util 


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
  def __init__(self, text_tokenizer, s2_tokenizer, data: pd.DataFrame, s2level: int, 
    unique_cells_df: pd.DataFrame, cellid_to_label: Dict[int, int], 
    dprob: util.DistanceProbability, is_dist: Boolean = False):

    self.text_tokenizer = text_tokenizer
    self.s2_tokenizer = s2_tokenizer

    self.is_dist = is_dist
    
    data = data.assign(point=data.end_point)


    data['cellid'] = data.point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))


    data['neighbor_cells'] = data.cellid.apply(
      lambda x: gutil.neighbor_cellid(x))

    if is_dist:
      dist_lists = data.start_point.apply(
        lambda start: calc_dist(start, unique_cells_df)
      )
      
      self.prob = dist_lists.swifter.apply(
        lambda row: [dprob(dist) for dist in row.values.tolist()], axis=1) 
      self.prob = self.prob.tolist()


    # Tokenize instructions.
    self.encodings = self.text_tokenizer(
      data.instructions.tolist(), truncation=True,
      padding=True, add_special_tokens=True)

    data['far_cells'] = data.cellid.apply(
      lambda cellid: unique_cells_df[unique_cells_df['cellid']==cellid].far.iloc[0])


    cellids_array = np.array(data.cellid.tolist())
    neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    far_cells_array = np.array(data.far_cells.tolist())

    self.points = data.point.apply(
      lambda x: gutil.tuple_from_point(x)).tolist()

    self.labels = data.cellid.apply(lambda x: cellid_to_label[x]).tolist()

    self.cellids = self.s2_tokenizer(cellids_array)
    self.neighbor_cells = self.s2_tokenizer(neighbor_cells_array)

    self.far_cells = self.s2_tokenizer(far_cells_array)

    if 'landmarks' in data:
      data['landmarks'] = data.landmarks.apply(
        lambda l: [gutil.cellid_from_point(x, s2level) for x in l])
      landmarks_array = np.array(data.landmarks.tolist())
      self.landmarks = self.s2_tokenizer(landmarks_array)
    else:
      self.landmarks = [0] * len(self.cellids)

    if 'route' in data:
      data['route'] = data.route.apply(
        lambda l: [gutil.cellid_from_point(x, s2level) for x in l])
      route_array = np.array(data.route.tolist())
      self.route = self.s2_tokenizer(route_array)
    else:
      self.route = [0] * len(self.cellids)

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
    cellid = self.cellids[idx]
    landmarks = self.landmarks[idx]
    route = self.route[idx]

    neighbor_cells = torch.tensor(self.neighbor_cells[idx])
    far_cells = torch.tensor(self.far_cells[idx])
    point = torch.tensor(self.points[idx])
    label = torch.tensor(self.labels[idx])
    if self.is_dist:
      prob = torch.tensor(self.prob[idx])
    else:
      prob = torch.tensor([])
    
    sample = {'text': text, 'cellid': cellid, 'neighbor_cells': neighbor_cells, 
      'far_cells': far_cells, 'point': point, 'label': label, 'prob': prob, 
      'landmarks': landmarks, 'route': route}

    return sample

  def __len__(self):
    return len(self.cellids)

def calc_dist(start, unique_cells_df):
  return unique_cells_df.swifter.apply(
    lambda end: gutil.get_distance_between_points(start, end.point), axis=1)


