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

import os
import pandas as pd
from sklearn.utils import shuffle

from cabby.geo import regions
from cabby.geo import util as gutil


class HumanDataset:
  def __init__(self, data_dir: str, s2level: int, region: str, lines: bool = True):
    self.train = self.load_data(data_dir, 'train', lines=lines)
    self.valid = self.load_data(data_dir, 'dev', lines=lines)
    self.test = self.load_data(data_dir, 'test', lines=lines)
    
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

    columns_keep = ds.columns.difference(
      ['instructions', 'end_point', 'start_point'])
    ds.drop(columns_keep, 1, inplace=True)

    ds = shuffle(ds)
    ds.reset_index(inplace=True, drop=True)
    return ds


class RUNDataset:
  def __init__(self, data_dir: str, s2level: int, lines: bool = False):
    train_ds, valid_ds, test_ds, ds = self.load_data(data_dir, lines=lines)

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


class RVSDataset:
  def __init__(self, data_dir: str, s2level: int, region: str, lines: bool = True):

    train_ds = self.load_data(data_dir, 'train', lines)
    valid_ds = self.load_data(data_dir, 'dev', lines)
    test_ds = self.load_data(data_dir, 'test', lines)

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
    # ds = pd.read_json(path_ds, lines=lines)
        
    import json
    lineByLine = []

    with open(path_ds, 'r') as f:
      db = json.load(f)
      lineByLine.append(db)
    
    for line in lineByLine:
      logging.info(f" ?????????? {type(line['geo_landmarks'])} ")

    to_json = json.dumps(lineByLine)
    ds = pd.read_json(to_json)


    logging.info(f"Size of dataset before removal of duplication: {ds.shape[0]}")
    # ds = pd.concat([ds.drop(['geo_landmarks'], axis=1), ds['geo_landmarks'].apply(pd.Series)], axis=1)
    logging.info(f"!!!!!!!!! {ds.geo_landmarks.keys()}")

    # logging.info(f"!!!!!!!!! {ds.apply(lambda x: type(x.geo_landmarks[0]), axis=1)}")

    ds['end_osmid'] = ds.geo_landmarks.apply(lambda x: x['end_point'][1])
    ds['start_osmid'] = ds.geo_landmarks.apply(lambda x: x['start_point'][1])
    ds['end_pivot'] = ds.geo_landmarks['end_point']
    ds['end_point'] = ds.geo_landmarks.apply(lambda x: x['end_point'][3])
    ds['start_point'] = ds.geo_landmarks.apply(lambda x: x['start_point'][3])
    ds = ds.drop_duplicates(subset=['end_osmid', 'start_osmid'], keep='last')
    return ds    


