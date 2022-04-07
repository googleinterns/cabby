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

from absl import logging

import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
import torch

from typing import Optional


from cabby.geo import regions
from cabby.geo import util as gutil
from cabby.model import dataset_item
from cabby.model import util 
from transformers import DistilBertTokenizerFast, T5Tokenizer


MODELS = ['Dual-Encoder-Bert', 'Classification-Bert', 'S2-Generation-T5', 'S2-Generation-T5-Landmarks']

T5_TYPE = "t5-small"
BERT_TYPE = 'distilbert-base-uncased'


# DISTRIBUTION_SCALE_DISTANCEA is a factor (in meters) that gives the overall 
# scale in meters for the distribution.
DISTRIBUTION_SCALE_DISTANCE = 1000
dprob = util.DistanceProbability(DISTRIBUTION_SCALE_DISTANCE)

tokenizerT5 = T5Tokenizer.from_pretrained(T5_TYPE)


class Dataset: 
  def __init__(self, data_dir: str, s2level: int, region: Optional[str], model_type: str):
    self.data_dir = data_dir
    self.s2level = s2level
    self.region = region
    self.model_type = model_type

    self.unique_cellid = {}
    self.cellid_to_label = {}
    self.label_to_cellid = {}

    self.train = None
    self.valid = None
    self.test = None

    self.set_tokenizers()

  def set_tokenizers(self):
    assert self.model_type in MODELS
    if self.model_type in ['Dual-Encoder-Bert', 'Classification-Bert']:      
      self.text_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_TYPE)
      self.s2_tokenizer = util.binary_representation
    elif 'S2-Generation-T5' in self.model_type:
      self.text_tokenizer = T5Tokenizer.from_pretrained(T5_TYPE)
      self.s2_tokenizer = self.tokenize_cell

  def tokenize_cell(self, list_cells):
    if isinstance(list_cells[0], list): 
      labels = []
      for c_list in list_cells:
        list_lables = []
        for c in c_list:
          list_lables.append(str(self.cellid_to_label[c]))

        labels.append('; '.join(list_lables))

    else:
      labels = [str(self.cellid_to_label[c]) for c in list_cells]

    return tokenizerT5(
      labels, return_tensors="pt", padding=True, truncation=True).input_ids
    

    
  def create_dataset(self, infer_only: bool = False
  ) -> dataset_item.TextGeoDataset:
    '''Loads data and creates datasets and train, validate and test sets.
    Returns:
      The train, validate and test sets and the dictionary of labels to cellids.
    '''

    points = gutil.get_centers_from_s2cellids(self.unique_cellid)

    unique_cells_df = pd.DataFrame(
      {'point': points, 'cellid': self.unique_cellid})
    
    unique_cells_df['far'] = unique_cells_df.point.swifter.apply(
        lambda x: gutil.far_cellid(x, unique_cells_df))

    vec_cells = self.s2_tokenizer(unique_cells_df.cellid.tolist())
    tens_cells = torch.tensor(vec_cells)

    # Create RUN dataset.
    train_dataset = None
    val_dataset = None
    logging.info("Starting to create the splits")
    if infer_only == False:
      train_dataset = dataset_item.TextGeoSplit(
        self.text_tokenizer,
        self.s2_tokenizer,
        self.train, self.s2level, unique_cells_df, 
        self.cellid_to_label, dprob)
      logging.info(
        f"Finished to create the train-set with {len(train_dataset)} samples")
      val_dataset = dataset_item.TextGeoSplit(
        self.text_tokenizer,
        self.s2_tokenizer, 
        self.valid, self.s2level, unique_cells_df, 
        self.cellid_to_label, dprob)
      logging.info(
        f"Finished to create the valid-set with {len(val_dataset)} samples")
    test_dataset = dataset_item.TextGeoSplit(
      self.text_tokenizer,
      self.s2_tokenizer,
      self.test, self.s2level, unique_cells_df, 
      self.cellid_to_label, dprob)
    logging.info(
      f"Finished to create the test-set with {len(test_dataset)} samples")

    return dataset_item.TextGeoDataset.from_TextGeoSplit(
      train_dataset, val_dataset, test_dataset, 
      np.array(self.unique_cellid), 
      tens_cells, self.label_to_cellid)


class HumanDataset(Dataset):
  def __init__(
    self, data_dir: str, 
    s2level: int, 
    region: Optional[str], 
    model_type: str = "Dual-Encoder-Bert"):

    Dataset.__init__(self, data_dir, s2level, region, model_type)
    self.train = self.load_data(data_dir, 'train', lines=True)
    self.valid = self.load_data(data_dir, 'dev', lines=True)
    self.test = self.load_data(data_dir, 'test', lines=True)
    
    # Get labels.
    active_region = regions.get_region(region)
    unique_cellid = gutil.cellids_from_polygon(active_region.polygon, s2level)
    label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellid)}
    cellid_to_label = {cellid: idx for idx, cellid in enumerate(unique_cellid)}

    self.unique_cellid = unique_cellid
    self.label_to_cellid = label_to_cellid
    self.cellid_to_label = cellid_to_label


  def load_data(self, data_dir: str, ds_set: str, lines: bool):

    ds_path = os.path.join(data_dir, ds_set + '.json')
    assert os.path.exists(ds_path), f"{ds_path} doesn't exsits"

    ds = pd.read_json(ds_path, lines=lines)
    ds['instructions'] = ds['content']
    ds['end_point'] = ds['rvs_goal_point'].apply(gutil.point_from_str_coord_xy)
    ds['start_point'] = ds['rvs_start_point'].apply(gutil.point_from_str_coord_xy)

    if 'landmarks' in ds:
      ds['landmarks'] = ds.landmarks.apply(self.process_landmarks)

    columns_keep = ds.columns.difference(
      ['instructions', 'end_point', 'start_point', 'landmarks'])
    ds.drop(columns_keep, 1, inplace=True)

    ds = shuffle(ds)
    ds.reset_index(inplace=True, drop=True)
    return ds

  def process_landmarks(self, landmarks_str_one_line):
    ladmarks_str_list = landmarks_str_one_line.split(';')
    return [gutil.point_from_str_coord_yx(
      landmark_str.split(':')[-1]) for landmark_str in ladmarks_str_list]



  

class RUNDataset(Dataset):
  def __init__(
    self, 
    data_dir: str, 
    s2level: int, 
    region: Optional[str], 
    model_type: str = "Dual-Encoder-Bert"):

    Dataset.__init__(self, data_dir, s2level, None, model_type)

    train_ds, valid_ds, test_ds, ds = self.load_data(data_dir, lines=False)

    # Get labels.
    map_1 = regions.get_region("RUN-map1")
    map_2 = regions.get_region("RUN-map2")
    map_3 = regions.get_region("RUN-map3")

    logging.info(map_1.polygon.wkt)
    logging.info(map_2.polygon.wkt)
    logging.info(map_3.polygon.wkt)

    unique_cellid_map_1 = gutil.cellids_from_polygon(map_1.polygon, s2level)
    unique_cellid_map_2 = gutil.cellids_from_polygon(map_2.polygon, s2level)
    unique_cellid_map_3 = gutil.cellids_from_polygon(map_3.polygon, s2level)

    unique_cellid = (
      unique_cellid_map_1 + unique_cellid_map_2 + unique_cellid_map_3)
    label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellid)}
    cellid_to_label = {cellid: idx for idx, cellid in enumerate(unique_cellid)}

    self.train = train_ds
    self.valid = valid_ds
    self.test = test_ds
    self.ds = ds
    self.unique_cellid = unique_cellid
    self.label_to_cellid = label_to_cellid
    self.cellid_to_label = cellid_to_label

  def load_data(self, data_dir: str, lines: bool):
    ds = pd.read_json(os.path.join(data_dir, 'dataset.json'), lines=lines)
    ds['instructions'] = ds.groupby(
      ['id'])['instruction'].transform(lambda x: ' '.join(x))

    ds = ds.drop_duplicates(subset='id', keep="last")

    columns_keep = ds.columns.difference(
      ['map', 'id', 'instructions', 'end_point', 'start_point'])
    ds.drop(columns_keep, 1, inplace=True)

    ds = shuffle(ds)
    ds.reset_index(inplace=True, drop=True)

    dataset_size = ds.shape[0]
    logging.info(f"Size of dataset: {ds.shape[0]}")
    train_size = round(dataset_size * 80 / 100)
    valid_size = round(dataset_size * 10 / 100)

    train_ds = ds.iloc[:train_size]
    valid_ds = ds.iloc[train_size:train_size + valid_size]
    test_ds = ds.iloc[train_size + valid_size:]
    return train_ds, valid_ds, test_ds, ds


class RVSDataset(Dataset):
  def __init__(
    self, data_dir: str, s2level: int, region: Optional[str], model_type: str = "RVS"):
    Dataset.__init__(self, data_dir, s2level, region, model_type)
    train_ds = self.load_data(data_dir, 'train', True)
    valid_ds = self.load_data(data_dir, 'dev', True)
    test_ds = self.load_data(data_dir, 'test', True)

    # Get labels.
    active_region = regions.get_region(region)
    unique_cellid = gutil.cellids_from_polygon(active_region.polygon, s2level)
    label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellid)}
    cellid_to_label = {cellid: idx for idx, cellid in enumerate(unique_cellid)}

    self.train = train_ds
    self.valid = valid_ds
    self.test = test_ds
    self.unique_cellid = unique_cellid
    self.label_to_cellid = label_to_cellid
    self.cellid_to_label = cellid_to_label


  def load_data(self, data_dir: str, split: str, lines: bool):
    path_ds= os.path.join(data_dir, f'ds_{split}.json')
    assert os.path.exists(path_ds), path_ds
    ds = pd.read_json(path_ds, lines=lines)
        
    logging.info(f"Size of dataset before removal of duplication: {ds.shape[0]}")

    ds = pd.concat([ds.drop(['geo_landmarks'], axis=1), ds['geo_landmarks'].apply(pd.Series)], axis=1)

    ds['end_osmid'] = ds.end_point.apply(lambda x: x[1])
    ds['start_osmid'] = ds.start_point.apply(lambda x: x[1])
    ds['end_pivot'] = ds.end_point
    ds['end_point'] = ds.end_point.apply(lambda x: gutil.point_from_list_coord(x[3]))
    ds['start_point'] = ds.start_point.apply(lambda x: gutil.point_from_list_coord(x[3]))
    ds = ds.drop_duplicates(subset=['end_osmid', 'start_osmid'], keep='last')
    return ds    


