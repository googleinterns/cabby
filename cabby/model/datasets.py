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
import sys

from absl import logging

from gensim.models import KeyedVectors
import numpy as np
import os
import pandas as pd
from random import sample
from sklearn.utils import shuffle
from s2geometry import pywraps2 as s2

import torch

from typing import Optional, Any

from cabby.geo import regions
from cabby.geo import util as gutil
from cabby.model import dataset_item
from cabby.model import util





FAR_DISTANCE_THRESHOLD = 2000 # Minimum distance between far cells in meters.

# DISTRIBUTION_SCALE_DISTANCEA is a factor (in meters) that gives the overall
# scale in meters for the distribution.
DISTRIBUTION_SCALE_DISTANCE = 1000
dprob = util.DistanceProbability(DISTRIBUTION_SCALE_DISTANCE)



class Dataset:
  def __init__(
    self,
    data_dir: str,
    s2level: int,
    train_region: Optional[str], 
    dev_region: Optional[str], 
    test_region: Optional[str], 
    model_type: str,
    n_fixed_points: int = 4,
    train_graph_embed_path: str = "", 
    dev_graph_embed_path: str = "", 
    test_graph_embed_path: str = ""
  ):
    self.data_dir = data_dir
    self.s2level = s2level
    self.train_region = train_region
    self.dev_region = dev_region
    self.test_region = test_region

    self.model_type = model_type
    self.n_fixed_points = n_fixed_points

    self.graph_embed_size = {}
    self.graph_embed_file = {}
    
    self.get_graph_embed(train_graph_embed_path, 'train')
    self.get_graph_embed(dev_graph_embed_path, 'dev')
    self.get_graph_embed(test_graph_embed_path, 'test')

    self.unique_cellids = {}
    self.cellid_to_label = {}
    self.label_to_cellid = {}

    self.train_raw = None
    self.dev_raw = None
    self.test_raw = None

  def get_graph_embed(self, graph_embed_path, set_type):
    if os.path.exists(graph_embed_path):
      self.graph_embed_file[set_type] = KeyedVectors.load_word2vec_format(graph_embed_path)
      first_cell = self.graph_embed_file[set_type].index_to_key[0]
      self.graph_embed_size[set_type] = self.graph_embed_file[set_type][first_cell].shape[0]

      logging.info(f"Dataset-{set_type} with graph embedding size {self.graph_embed_size[set_type]}")
    
    else:
      self.graph_embed_file[set_type] = None
      self.graph_embed_size[set_type] = 0


  def process_route(self, row):
    route_str = row.route
    route_str = route_str.replace('LINESTRING', "").replace('(', "").replace(')', "")
    landmarks_str_list = route_str.split(',')

    route = [
      gutil.point_from_str_coord_xy(landmark_str) for landmark_str in reversed(landmarks_str_list)]

    route.insert(0, row['end_point'])

    assert route[0] == row['end_point'], f"end: {row['end_point']} route: {route[0]} - {route[-1]}"
    return route

  def process_landmarks(self, row):
    points = [row['end_point'], row['near_pivot'], row['main_pivot'], row['start_point']]

    assert points[0] == row['end_point']
    return points

  def get_specific_landmark(self, landmarks_str_one_line, landmark_name):
    return gutil.point_from_list_coord_yx(landmarks_str_one_line[landmark_name][0])

  
  def set_cells(self, region, s2level, set_type):
    active_region = regions.get_region(region)
    self.unique_cellids[set_type] = gutil.cellids_from_polygon(active_region.polygon, s2level)
    self.label_to_cellid[set_type] = {
      idx: cellid for idx, cellid in enumerate(self.unique_cellids[set_type])}
    self.cellid_to_label[set_type] = {
      cellid: idx for idx, cellid in enumerate(self.unique_cellids[set_type])}

  def create_dataset(
    self, infer_only: bool = False, is_dist: bool = False,
    far_cell_dist: int = FAR_DISTANCE_THRESHOLD
                     ): # -> dataset_item.TextGeoDataset:
    '''Loads data and creates datasets and train, validate and test sets.
    Returns:
      The train, validate and test sets and the dictionary of labels to cellids.
    '''

    # Create RUN dataset.

    logging.info("Starting to create the splits")
    if infer_only == False:
      train_set = dataset_item.TextGeoSplit(
        set_type='train',
        unique_cellids = self.unique_cellids['train'],
        data=self.train_raw, 
        s2level=self.s2level, 
        cellid_to_label=self.cellid_to_label['train'], 
        model_type=self.model_type,
        dprob=dprob, 
        is_dist=is_dist,
        far_cell_dist = far_cell_dist,
        graph_embed_file=self.graph_embed_file['train'],
        graph_embed_size=self.graph_embed_size['train'])
      logging.info(
        f"Finished to create the train-set with {len(train_set)} samples")
      
      dev_set = dataset_item.TextGeoSplit(
        set_type='dev',
        unique_cellids = self.unique_cellids['dev'],
        data=self.dev_raw, 
        s2level=self.s2level, 
        cellid_to_label=self.cellid_to_label['dev'], 
        model_type=self.model_type,
        dprob=dprob, 
        is_dist=is_dist,
        far_cell_dist = far_cell_dist,
        graph_embed_file=self.graph_embed_file['dev'],
        graph_embed_size = self.graph_embed_size['dev'])
      logging.info(
        f"Finished to create the dev-set with {len(dev_set)} samples")

    test_set = dataset_item.TextGeoSplit(
      set_type='test',
      unique_cellids = self.unique_cellids['test'],
      data=self.test_raw, 
      s2level=self.s2level, 
      cellid_to_label=self.cellid_to_label['test'], 
      model_type=self.model_type,
      dprob=dprob, 
      is_dist=is_dist,
      far_cell_dist = far_cell_dist,
      graph_embed_file=self.graph_embed_file['test'],
      graph_embed_size = self.graph_embed_size['test'])
    logging.info(
      f"Finished to create the test-set with {len(test_set)} samples")
    return dataset_item.TextGeoDataset.from_TextGeoSplit(train_set=train_set, dev_set=dev_set, test_set=test_set)

  def get_fixed_point_along_route(self, row):
    points_list = row.route
    end_point = row.end_point
    avg_number = round(len(points_list) / self.n_fixed_points)
    fixed_points = []
    for i in range(self.n_fixed_points):
      curr_point = points_list[i * avg_number]
      fixed_points.append(curr_point)

    assert len(fixed_points) == self.n_fixed_points

    fixed_points.append(row.end_point)
    reversed_points = list(reversed(fixed_points))
    assert reversed_points[0] == end_point
    return reversed_points


class HumanDataset(Dataset):
  def __init__(
    self, data_dir: str,
    s2level: int,
    train_region: Optional[str],
    dev_region: Optional[str],
    test_region: Optional[str],
    n_fixed_points: int = 4,
    train_graph_embed_path: str = "",
    dev_graph_embed_path: str = "",
    test_graph_embed_path: str = "",
    model_type: str = "Dual-Encoder-Bert"):

    Dataset.__init__(
      self, data_dir, s2level, train_region, dev_region, test_region, 
      model_type, n_fixed_points, train_graph_embed_path, dev_graph_embed_path, test_graph_embed_path)
    self.train_raw = self.load_data(data_dir, 'train', lines=True)
    self.dev_raw = self.load_data(data_dir, 'dev', lines=True)
    self.test_raw = self.load_data(data_dir, 'test', lines=True)

    self.unique_cellid = {}
    self.label_to_cellid = {}
    self.cellid_to_label = {}

    self.set_cells(train_region, s2level, 'train')
    self.set_cells(dev_region, s2level, 'dev')
    self.set_cells(test_region, s2level, 'test')



  def load_data(self, data_dir: str, ds_set: str, lines: bool):

    ds_path = os.path.join(data_dir, ds_set + '.json')
    assert os.path.exists(ds_path), f"{ds_path} doesn't exsits"

    ds = pd.read_json(ds_path, lines=lines)
    ds['instructions'] = ds['content']
    ds['end_point'] = ds['rvs_goal_point'].apply(gutil.point_from_list_coord_yx)
    ds['start_point'] = ds['rvs_start_point'].apply(gutil.point_from_list_coord_yx)

    if 'landmarks' in ds:
      ds['near_pivot'] = ds.landmarks.apply(
        lambda x: self.get_specific_landmark(x, 'near_pivot'))
      ds['main_pivot'] = ds.landmarks.apply(
        lambda x: self.get_specific_landmark(x, 'main_pivot'))

      ds['landmarks'] = ds.apply(self.process_landmarks, axis=1)

    if 'route' in ds:
      ds['route'] = ds.apply(self.process_route, axis=1)
      ds['route_fixed'] = ds.apply(self.get_fixed_point_along_route, axis=1)

      ds['start_end'] = ds.apply(self.get_fixed_point_along_route, axis=1)

    ds = shuffle(ds)
    ds.reset_index(inplace=True, drop=True)
    return ds


class RUNDataset(Dataset):
  def __init__(
    self,
    data_dir: str,
    s2level: int,
    region: Optional[str],
    graph_embed_path: str = "",
    n_fixed_points: int = 4,
    model_type: str = "Dual-Encoder-Bert"):
    Dataset.__init__(
      self, data_dir, s2level, None, model_type, n_fixed_points, graph_embed_path
    )

    train_ds, dev_ds, test_ds, ds = self.load_data(data_dir, lines=False)

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
    self.dev = dev_ds
    self.test = test_ds
    self.ds = ds
    self.unique_cellid = unique_cellid
    self.label_to_cellid = label_to_cellid
    self.cellid_to_label = cellid_to_label

  def load_data(self, data_dir: str, lines: bool):
    ds = pd.read_json(os.path.join(data_dir, 'dataset.json'), lines=lines)
    ds['instructions'] = ds.groupby(
      ['id'])['instruction'].transform(lambda x: ' '.join(x))

    ds = ds.drop_duplicates(subset='id', keep="last", ignore_index=True)

    columns_keep = ds.columns.difference(
      ['map', 'id', 'instructions', 'end_point', 'start_point'])
    ds.drop(columns_keep, 1, inplace=True)

    ds = shuffle(ds)
    ds.reset_index(inplace=True, drop=True)

    dataset_size = ds.shape[0]
    logging.info(f"Size of dataset: {ds.shape[0]}")
    train_size = round(dataset_size * 80 / 100)
    dev_size = round(dataset_size * 10 / 100)

    train_ds = ds.iloc[:train_size]
    dev_ds = ds.iloc[train_size:train_size + dev_size]
    test_ds = ds.iloc[train_size + dev_size:]
    return train_ds, dev_ds, test_ds, ds


class RVSDataset(Dataset):
  def __init__(
    self, data_dir: str,
    s2level: int,
    train_region: Optional[str],
    dev_region: Optional[str],
    test_region: Optional[str],
    n_fixed_points: int = 4,
    train_graph_embed_path: str = "",
    dev_graph_embed_path: str = "",
    test_graph_embed_path: str = "",
    model_type: str = "Dual-Encoder-Bert"):

    Dataset.__init__(
      self, data_dir, s2level, train_region, dev_region, test_region, 
      model_type, n_fixed_points, train_graph_embed_path, dev_graph_embed_path, test_graph_embed_path)
    self.train_raw = self.load_data(data_dir, 'train', True)
    self.dev_raw = self.load_data(data_dir, 'dev', True)
    self.test_raw = self.load_data(data_dir, 'test', True)

    self.unique_cellid = {}
    self.label_to_cellid = {}
    self.cellid_to_label = {}

    # Get labels.

    self.set_cells(train_region, s2level, 'train')
    self.set_cells(dev_region, s2level, 'dev')
    self.set_cells(test_region, s2level, 'test')


  def load_data(self, data_dir: str, split: str, lines: bool):
    path_ds = os.path.join(data_dir, f'ds_{split}.json')
    assert os.path.exists(path_ds), path_ds
    ds = pd.read_json(path_ds, lines=lines)

    if 'geo_landmarks' in ds:
      ds['landmarks'] = ds.apply(self.process_landmarks, axis=1)
      ds['landmarks_ner'] = ds.geo_landmarks.apply(self.process_landmarks_ner)
      ds['landmarks_ner_and_point'] = ds.geo_landmarks.apply(self.process_landmarks_ner_single)

    ds = pd.concat(
      [ds.drop(['geo_landmarks'], axis=1), ds['geo_landmarks'].apply(pd.Series)], axis=1)

    ds['end_osmid'] = ds.end_point.apply(lambda x: x[1])
    ds['start_osmid'] = ds.start_point.apply(lambda x: x[1])
    ds['end_pivot'] = ds.end_point
    ds['end_point'] = ds.end_point.apply(lambda x: gutil.point_from_list_coord_yx(x[3]))
    ds['start_point'] = ds.start_point.apply(lambda x: gutil.point_from_list_coord_yx(x[3]))

    if 'route' in ds:
      ds['route'] = ds.apply(self.process_route, axis=1)
      ds['route_fixed'] = ds.apply(self.get_fixed_point_along_route, axis=1)

    logging.info(f"Size of dataset before removal of duplication: {ds.shape[0]}")

    ds = ds.drop_duplicates(subset=['end_osmid', 'start_osmid'], keep='last', ignore_index=True)

    logging.info(f"Size of dataset after removal of duplication: {ds.shape[0]}")

    ds = ds.reset_index(drop=True)
    return ds

  def process_landmarks(self, row):
    landmarks_dict = row.geo_landmarks
    landmarks_coords = [
      landmarks_dict['end_point'][-1],
      landmarks_dict['near_pivot'][-1],
      landmarks_dict['main_pivot'][-1],
      landmarks_dict['start_point'][-1],
    ]
    points = [gutil.point_from_list_coord_yx(
      coord) for coord in landmarks_coords]

    assert landmarks_coords[0] == landmarks_dict['end_point'][-1]

    return points

  def process_landmarks_ner(self, landmarks_dict):
    main_pivot = landmarks_dict['main_pivot'][2]
    near_pivot = landmarks_dict['near_pivot'][2]
    end_point = landmarks_dict['end_point'][2]

    landmarks_list = list(set([end_point, near_pivot, main_pivot]))

    landmarks_ner_list = [l for l in landmarks_list if l != 'None']
    return '; '.join(landmarks_ner_list)

  def process_landmarks_ner_single(self, landmarks_dict):
    main_pivot = landmarks_dict['main_pivot'][2]
    near_pivot = landmarks_dict['near_pivot'][2]
    end_point = landmarks_dict['end_point'][2]
    start_point = landmarks_dict['start_point'][2]

    main_pivot_point = landmarks_dict['main_pivot'][-1]
    near_pivot_point = landmarks_dict['near_pivot'][-1]
    end_point_point = landmarks_dict['end_point'][-1]
    start_point_point = landmarks_dict['start_point'][-1]

    landmarks_list = [
      (main_pivot, gutil.point_from_list_coord_yx(main_pivot_point)),
      (near_pivot, gutil.point_from_list_coord_yx(near_pivot_point)),
      (end_point, gutil.point_from_list_coord_yx(end_point_point)),
      (start_point, gutil.point_from_list_coord_yx(start_point_point))]

    return sample(landmarks_list, 1)[0]

  def process_route(self, row):
    route_list = row.route
    end_point = row.end_point
    route = [
      gutil.point_from_list_coord_xy(landmark) for landmark in reversed(route_list)]
    assert route[0] == end_point
    return route
