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
from random import sample
from sklearn.utils import shuffle
import torch


from typing import Optional


from cabby.geo import regions
from cabby.geo import util as gutil
from cabby.model import dataset_item
from cabby.model import util 
from transformers import DistilBertTokenizerFast, T5Tokenizer


MODELS = [
  'Dual-Encoder-Bert', 
  'Classification-Bert', 
  'S2-Generation-T5', 
  'S2-Generation-T5-Landmarks', 
  'S2-Generation-T5-Warmup-start-end', 
  'Text-2-Landmarks-NER-Generation-T5-Warmup', 
  'Landmarks-NER-2-S2-Generation-T5-Warmup', 
  'S2-Generation-T5-Path',
  ]

T5_TYPE = "t5-small"
BERT_TYPE = 'distilbert-base-uncased'


# DISTRIBUTION_SCALE_DISTANCEA is a factor (in meters) that gives the overall 
# scale in meters for the distribution.
DISTRIBUTION_SCALE_DISTANCE = 1000
dprob = util.DistanceProbability(DISTRIBUTION_SCALE_DISTANCE)

tokenizerT5 = T5Tokenizer.from_pretrained(T5_TYPE)


class Dataset: 
  def __init__(
    self, 
    data_dir: str, 
    s2level: int, 
    region: Optional[str], 
    model_type: str,
    n_fixed_points: int
    ):
    self.data_dir = data_dir
    self.s2level = s2level
    self.region = region
    self.model_type = model_type
    self.n_fixed_points = n_fixed_points

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
    elif 'T5' in self.model_type:
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
      labels = [str(util.get_valid_label(self.cellid_to_label,c)) for c in list_cells]

    return tokenizerT5(
      labels, padding=True, truncation=True).input_ids
    

  def process_route(self, route_str):
    route_str = route_str.replace('LINESTRING', "").replace('(', "").replace(')', "")
    landmarks_str_list = route_str.split(',')
    return [
      gutil.point_from_str_coord_xy(landmark_str) for landmark_str in landmarks_str_list]


  def process_landmarks(self, row):
    points = [row['end_point'], row['start_point'], row['main_pivot'], row['near_pivot']]
    return points
  
  def get_specific_landmark(self, landmarks_str_one_line, landmark_name):

    landmarks_str_list = landmarks_str_one_line.split(';')

    landmark_found = None
    for landmark_str in landmarks_str_list:
      if landmark_name in landmark_str:
        landmark_found = gutil.point_from_str_coord_yx(landmark_str.split(':')[-1])

    return landmark_found

    
  def create_dataset(self, infer_only: bool = False,
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
        self.cellid_to_label, self.model_type, dprob)
      logging.info(
        f"Finished to create the train-set with {len(train_dataset)} samples")
      val_dataset = dataset_item.TextGeoSplit(
        self.text_tokenizer,
        self.s2_tokenizer, 
        self.valid, self.s2level, unique_cells_df, 
        self.cellid_to_label, self.model_type, dprob)
      logging.info(
        f"Finished to create the valid-set with {len(val_dataset)} samples")
    test_dataset = dataset_item.TextGeoSplit(
      self.text_tokenizer,
      self.s2_tokenizer,
      self.test, self.s2level, unique_cells_df, 
      self.cellid_to_label, self.model_type, dprob)
    logging.info(
      f"Finished to create the test-set with {len(test_dataset)} samples")

    return dataset_item.TextGeoDataset.from_TextGeoSplit(
      train_dataset, val_dataset, test_dataset, 
      np.array(self.unique_cellid), 
      tens_cells, self.label_to_cellid)


  def get_fixed_point_along_route(self, points_list):
    avg_number = round(len(points_list)/self.n_fixed_points)
    fixed_points = []
    for i in range(self.n_fixed_points):
      curr_point = points_list[i*avg_number]
      fixed_points.append(curr_point)
    
    assert len(fixed_points) == self.n_fixed_points
    return fixed_points


class HumanDataset(Dataset):
  def __init__(
    self, data_dir: str, 
    s2level: int, 
    region: Optional[str], 
    n_fixed_points: int,
    model_type: str = "Dual-Encoder-Bert"):

    Dataset.__init__(self, data_dir, s2level, region, model_type, n_fixed_points)
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
      ds['near_pivot'] = ds.landmarks.apply(
        lambda x: self.get_specific_landmark(x, 'near_pivot'))
      ds['main_pivot'] = ds.landmarks.apply(
        lambda x: self.get_specific_landmark(x, 'main_pivot'))
    
      ds['landmarks'] = ds.apply(self.process_landmarks, axis=1)
    
    if 'route' in ds:
      ds['route'] = ds.route.apply(self.process_route)
      ds['route_fixed'] = ds.route.apply(self.get_fixed_point_along_route)
    
      ds['start_end'] = ds.route.apply(self.get_fixed_point_along_route)
    columns_keep = ds.columns.difference(
      [
        'instructions', 
        'end_point', 
        'start_point', 
        'landmarks', 
        'route', 
        'near_pivot',
        'main_pivot',
        'route_fixed'])
    ds.drop(columns_keep, 1, inplace=True)

    ds = shuffle(ds)
    ds.reset_index(inplace=True, drop=True)
    return ds


class RUNDataset(Dataset):
  def __init__(
    self, 
    data_dir: str, 
    s2level: int, 
    region: Optional[str], 
    n_fixed_points: int = 4,
    model_type: str = "Dual-Encoder-Bert"):

    Dataset.__init__(
      self, data_dir, s2level, None, model_type, n_fixed_points
)

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
    self, 
    data_dir: str, 
    s2level: int, 
    region: Optional[str], 
    n_fixed_points: int,
    model_type: str = "Dual-Encoder-Bert"):
    Dataset.__init__(
      self, data_dir, s2level, region, model_type, n_fixed_points)
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
            
    if 'geo_landmarks' in ds:
      ds['landmarks'] =  ds.geo_landmarks.apply(self.process_landmarks)
      ds['landmarks_ner'] =  ds.geo_landmarks.apply(self.process_landmarks_ner)
      ds['landmarks_ner_and_point'] =  ds.geo_landmarks.apply(self.process_landmarks_ner_single)


    if 'route' in ds: 
      ds['route'] = ds.route.apply(self.process_route)
      ds['route_fixed'] = ds.route.apply(self.get_fixed_point_along_route)

    ds = pd.concat([ds.drop(['geo_landmarks'], axis=1), ds['geo_landmarks'].apply(pd.Series)], axis=1)

    ds['end_osmid'] = ds.end_point.apply(lambda x: x[1])
    ds['start_osmid'] = ds.start_point.apply(lambda x: x[1])
    ds['end_pivot'] = ds.end_point
    ds['end_point'] = ds.end_point.apply(lambda x: gutil.point_from_list_coord_yx(x[3]))
    ds['start_point'] = ds.start_point.apply(lambda x: gutil.point_from_list_coord_yx(x[3]))    

    logging.info(f"Size of dataset before removal of duplication: {ds.shape[0]}")

    ds = ds.drop_duplicates(subset=['end_osmid', 'start_osmid'], keep='last')

    logging.info(f"Size of dataset after removal of duplication: {ds.shape[0]}")


    return ds    

  def process_landmarks(self, landmarks_dict):
    landmarks_corrds = [
      landmarks_dict['end_point'][-1], 
      landmarks_dict['start_point'][-1],
      landmarks_dict['main_pivot'][-1],
      landmarks_dict['near_pivot'][-1]]
    points = [gutil.point_from_list_coord_yx(
      coord) for coord in landmarks_corrds]
    return points


  def process_landmarks_ner(self, landmarks_dict):
    main_pivot = landmarks_dict['main_pivot'][2]
    near_pivot = landmarks_dict['near_pivot'][2]
    end_point = landmarks_dict['end_point'][2]

    
    landmarks_list = list(set([main_pivot, near_pivot, end_point]))

    landmarks_ner_list = [l for l in landmarks_list if l!='None']
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
    
  def process_route(self, route_list):
    return [
      gutil.point_from_list_coord_xy(landmark) for landmark in route_list]
  


