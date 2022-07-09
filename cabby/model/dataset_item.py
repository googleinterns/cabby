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

import mapply

import attr

from cabby.geo import util as gutil
from cabby.model import util

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mapply.init(
  n_workers=-1,
  progressbar=True,
)


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
  def load(cls, dataset_dir: Text, model_type: Text,
           s2_level: Text, unique_cellid_path: Text, tensor_cellid_path: Text,
           label_to_cellid_path: Text):
    dataset_model_path = os.path.join(dataset_dir, str(model_type))
    dataset_path = os.path.join(dataset_model_path, str(s2_level))

    train_path_dataset = os.path.join(dataset_path, 'train.pth')
    valid_path_dataset = os.path.join(dataset_path, 'valid.pth')
    test_path_dataset = os.path.join(dataset_path, 'test.pth')
    unique_cellid_path = os.path.join(dataset_path, "unique_cellid.npy")
    tensor_cellid_path = os.path.join(dataset_path, "tensor_cellid.pth")
    label_to_cellid_path = os.path.join(dataset_path, "label_to_cellid.npy")

    logging.info("Loading dataset from <== {}.".format(dataset_path))
    train_dataset = torch.load(train_path_dataset)
    valid_dataset = torch.load(valid_path_dataset)
    test_dataset = torch.load(test_path_dataset)
    logging.info(f"Size of train set: {len(train_dataset)}" +
                 f", Size of validation set: {len(valid_dataset)}, Size of test set: {len(test_dataset)}")

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
           test_path_dataset: Text, unique_cellid_path: Text,
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
               model_type: str, dprob: util.DistanceProbability, dist_matrix: pd.DataFrame,
               graph_embed_file:Any = None, is_dist: Boolean = False
               ):

    self.text_tokenizer = text_tokenizer
    self.s2_tokenizer = s2_tokenizer
    self.cellid_to_label = cellid_to_label
    self.s2level = s2level
    self.is_dist = is_dist
    self.model_type = model_type
    self.graph_embed_file = graph_embed_file

    data = data.assign(end_point=data.end_point)

    data['cellid'] = data.end_point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))

    data['neighbor_cells'] = data.cellid.apply(
      lambda x: gutil.neighbor_cellid(x))

    # Tokenize instructions.

    instruction_list = data.instructions.tolist()
    if 'T5' in model_type:
      # Add prompt
      data.instructions = [model_type + ": " + t for t in instruction_list]
      logging.info(data.instructions.iloc[0])

    self.encodings = self.text_tokenizer(
      data.instructions.tolist(), truncation=True,
      padding=True, add_special_tokens=True)

    data['far_cells'] = data.cellid.apply(
      lambda cellid: unique_cells_df[unique_cells_df['cellid'] == cellid].far.iloc[0])

    cellids_array = np.array(data.cellid.tolist())
    neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    far_cells_array = np.array(data.far_cells.tolist())

    self.end_point = data.end_point.apply(
      lambda x: gutil.tuple_from_point(x)).tolist()

    self.labels = data.cellid.apply(lambda x: cellid_to_label[x]).tolist()

    self.start_point_cells = data.start_point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))

    if is_dist:
      logging.info(f"Calculating distances between {dist_matrix.shape[0]} cells")
      dist_lists = self.start_point_cells.apply(lambda start: self.calc_dist(start, dist_matrix))

      self.prob = dist_lists.mapply(
        lambda row: [dprob(dist) for dist in row.tolist()])

      self.prob = self.prob.tolist()

    self.start_point_labels = self.get_cell_to_lablel(self.start_point_cells.tolist())

    self.cellids = self.s2_tokenizer(cellids_array)

    self.neighbor_cells = self.s2_tokenizer(neighbor_cells_array)

    self.far_cells = self.s2_tokenizer(far_cells_array)

    self.set_S2_Generation_start_text_input(data)

    # Landmarks
    self.set_S2_Generation_T5_Landmarks(data)

    self.set_Landmarks_NER_2_S2_Generation_T5_Warmup(data)

    self.set_Text_2_Landmarks_NER_Generation_T5_Warmup(data)

    self.set_S2_Generation_T5_Path(data)

    self.set_S2_Generation_T5_Warmup_start_end(data)

    if graph_embed_file:

      self.graph_embed_end = data['cellid'].apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))
      self.graph_embed_start = self.start_point_cells.apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))

      self.graph_embed_landmarks = data.landmarks.apply(
        lambda l: [util.get_valid_graph_embed(
          self.graph_embed_file, str(cell)) for cell in l])

      self.set_S2_Generation_T5_Warmup_cell_embed_to_cell_label(data)
      self.set_S2_Generation_start_embedding_text_input(data)
      self.set_S2_Generation_T5_Landmarks_Embedding(data)

    else:

      default_empty = self.text_tokenizer(
        ['']*len(self.cellids), truncation=True, padding=True, add_special_tokens=True)

      self.graph_embed_start_and_prompt = default_empty

      self.start_text_and_prompt = default_empty

      self.start_embedding_text_and_prompt = default_empty

      self.landmarks_embed = default_empty.input_ids


    del self.graph_embed_file
    del self.start_point_cells
    del self.s2_tokenizer
    del self.text_tokenizer

  def get_cell_to_lablel(self, list_cells):
    if isinstance(list_cells[0], list):
      labels = []
      for c_list in list_cells:
        list_lables = []
        for c in c_list:
          list_lables.append(str(self.cellid_to_label[c]))

        labels.append('; '.join(list_lables))

    else:
      labels = [str(util.get_valid_cell_label(self.cellid_to_label, int(c))) for c in list_cells]

    return labels

  def set_S2_Generation_start_embedding_text_input(self, data):

    start_text_input_list = [
      str(i).replace(':', f': Start at {str(s)}.') for s, i in zip(
        self.graph_embed_start, data.instructions.tolist())]

    self.print_sample(
      mode_expected='S2-Generation-T5-start-embedding-text-input',
      input=start_text_input_list[0],
      output=self.labels[0])

    self.start_embedding_text_and_prompt = self.text_tokenizer(
      start_text_input_list, truncation=True, padding=True, add_special_tokens=True)


  def set_S2_Generation_start_text_input(self, data):

    start_text_input_list = [
      str(i).replace(':', f': Start at {str(s)}.') for s, i in zip(
        self.start_point_labels, data.instructions.tolist())]

    self.print_sample(
      mode_expected='S2-Generation-T5-start-text-input',
      input=start_text_input_list[0],
      output=self.labels[0])

    self.start_text_and_prompt = self.text_tokenizer(
      start_text_input_list, truncation=True, padding=True, add_special_tokens=True)

  def set_S2_Generation_T5_Landmarks(self, data):
    if 'T5' in self.model_type and 'landmarks' in data:
      data['landmarks'] = data.landmarks.apply(
        lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

      landmark_label = self.get_cell_to_lablel(data['landmarks'].tolist())

      self.print_sample(
        mode_expected='S2-Generation-T5-Landmarks',
        input=data.instructions.tolist()[0],
        output=landmark_label[0])

      self.landmarks = self.text_tokenizer(
        landmark_label, truncation=True, padding=True, add_special_tokens=True).input_ids

    else:
      self.landmarks = [0] * len(self.cellids)
      logging.warning("Landmarks not processed")


  def set_S2_Generation_T5_Landmarks_Embedding(self, data):

    landmarks_embed = [
      f"{end}; {landmarks}" for landmarks, end in zip(self.graph_embed_landmarks, self.labels)]

    self.print_sample(
      mode_expected='S2-Generation-T5-Landmarks',
      input=data.instructions.tolist()[0],
      output=landmarks_embed[0])

    self.landmarks_embed = self.text_tokenizer(
      landmarks_embed, truncation=True, padding=True, add_special_tokens=True).input_ids


  def set_Landmarks_NER_2_S2_Generation_T5_Warmup(self, data):
    if not ('T5' in self.model_type and 'landmarks_ner_and_point' in data):
      data = data.assign(landmarks_ner_and_prompt_input='')

      self.landmark_s2cell = [0] * len(self.cellids)


    else:

      landmarks_ner_input = [
        f"{self.model_type}: {ner}" for ner, point in data.landmarks_ner_and_point.tolist()]

      data = data.assign(landmarks_ner_and_prompt_input=landmarks_ner_input)

      landmark_cells = [
        gutil.cellid_from_point(
          point, self.s2level) for ner, point in data.landmarks_ner_and_point.tolist()]

      landmark_label = self.get_cell_to_lablel(landmark_cells)

      self.print_sample(
        mode_expected='Landmarks-NER-2-S2-Generation-T5-Warmup',
        input=data.landmarks_ner_and_prompt_input.tolist()[0],
        output=landmark_label[0])

      self.landmark_s2cell = self.text_tokenizer(
        landmark_label, truncation=True, padding=True, add_special_tokens=True).input_ids

    self.landmarks_ner_and_prompt_input = self.text_tokenizer(
      data.landmarks_ner_and_prompt_input.tolist(), truncation=True, padding=True, add_special_tokens=True)

  def set_Text_2_Landmarks_NER_Generation_T5_Warmup(self, data):
    if not 'landmarks_ner' in data:
      data = data.assign(landmarks_ner='')
      logging.warning("Landmarks NER not processed")

    self.print_sample(
      mode_expected='Text-2-Landmarks-NER-Generation-T5-Warmup',
      input=data.instructions.tolist()[0],
      output=data.landmarks_ner.tolist()[0])

    self.landmarks_ner = self.text_tokenizer(
      data.landmarks_ner.tolist(), truncation=True,
      padding=True, add_special_tokens=True).input_ids

  def set_S2_Generation_T5_Warmup_start_end(self, data):

    if 'T5' in self.model_type and 'route' in data:

      data['route_fixed'] = data.route_fixed.apply(
        lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

      route_fixed_label = self.get_cell_to_lablel(data.route_fixed.tolist())

      self.route_fixed = self.text_tokenizer(
        route_fixed_label, truncation=True, padding=True, add_special_tokens=True).input_ids


      start_end_point_list = [
        f"{self.model_type}: {str(e)}, {str(s)}" for s, e in zip(
          self.start_point_labels, self.labels)]

      self.print_sample(
        mode_expected='S2-Generation-T5-Warmup-start-end',
        input=start_end_point_list[0],
        output=route_fixed_label[0])

      self.start_end_and_prompt = self.text_tokenizer(
        start_end_point_list, truncation=True, padding=True, add_special_tokens=True)

    else:
      self.route = [0] * len(self.cellids)
      logging.warning("Route not processed")

  def set_S2_Generation_T5_Path(self, data):

    if 'T5' in self.model_type and 'route' in data:

      data['route'] = data.route.apply(
        lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

      route_label = self.get_cell_to_lablel(data.route.tolist())

      self.route = self.text_tokenizer(
        route_label, truncation=True, padding=True, add_special_tokens=True).input_ids

      self.print_sample(
        mode_expected='S2-Generation-T5-Path',
        input=data.instructions.tolist()[0],
        output=route_label[0])
    else:
      self.route_fixed = [0] * len(self.cellids)
      self.start_end_and_prompt = {
        'attention_mask': [0] * len(self.cellids),
        'input_ids': [0] * len(self.cellids)}

  def set_S2_Generation_T5_Warmup_cell_embed_to_cell_label(self, data):

    graph_embed_end_and_prompt = [
      f"{self.model_type}: {str(e)}" for e in self.graph_embed_end
    ]

    self.print_sample(
      mode_expected='S2-Generation-T5-Warmup-cell-embed-to-cell-label',
      input=graph_embed_end_and_prompt[0],
      output=self.labels[0])

    self.graph_embed_start_and_prompt = self.text_tokenizer(
      graph_embed_end_and_prompt, truncation=True, padding=True, add_special_tokens=True)



  def print_sample(self, mode_expected, input, output):
    if self.model_type == mode_expected:
      logging.info(
        f"\n Example {self.model_type}: \n" +
        f"  Input: '{input}'\n" +
        f"  Output: {output}\n" +
        f"  Goal: {self.labels[0]}\n" +
        f"  Start: {self.start_point_labels[0]}"
      )

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
    landmarks = torch.tensor(self.landmarks[idx])
    landmarks_ner = torch.tensor(self.landmarks_ner[idx])

    route = torch.tensor(self.route[idx])
    route_fixed = torch.tensor(self.route_fixed[idx])

    landmark_s2cell = torch.tensor(self.landmark_s2cell[idx])

    start_end_and_prompt = {key: torch.tensor(val[idx])
                            for key, val in self.start_end_and_prompt.items()}

    start_text_and_prompt = {key: torch.tensor(val[idx])
                            for key, val in self.start_text_and_prompt.items()}

    start_embedding_text_and_prompt = {key: torch.tensor(val[idx])
                            for key, val in self.start_embedding_text_and_prompt.items()}

    landmarks_ner_and_prompt_input = {
      key: torch.tensor(val[idx])
      for key, val in self.landmarks_ner_and_prompt_input.items()}

    graph_embed_start_and_prompt = {
      key: torch.tensor(val[idx])
      for key, val in self.graph_embed_start_and_prompt.items()}

    landmarks_embed = torch.tensor(self.landmarks_embed[idx])

    neighbor_cells = torch.tensor(self.neighbor_cells[idx])
    far_cells = torch.tensor(self.far_cells[idx])
    end_point = torch.tensor(self.end_point[idx])
    label = torch.tensor(self.labels[idx])
    if self.is_dist:
      prob = torch.tensor(self.prob[idx])
    else:
      prob = torch.tensor([])

    sample = {'text': text, 'cellid': cellid, 'neighbor_cells': neighbor_cells,
              'far_cells': far_cells, 'end_point': end_point, 'label': label, 'prob': prob,
              'landmarks': landmarks, 'route': route, 'route_fixed': route_fixed,
              'start_end_and_prompt_input_ids': start_end_and_prompt['input_ids'],
              'start_end_and_prompt_attention_mask': start_end_and_prompt['attention_mask'],
              'landmarks_ner': landmarks_ner, 'landmark_s2cell': landmark_s2cell,
              'landmarks_ner_and_prompt_input_ids': landmarks_ner_and_prompt_input['input_ids'],
              'landmarks_ner_and_prompt_input_attention': landmarks_ner_and_prompt_input['attention_mask'],
              'start_text_and_prompt_ids': start_text_and_prompt['input_ids'],
              'start_text_and_prompt_attention': start_text_and_prompt['attention_mask'],
              'start_embedding_text_and_prompt_ids': start_embedding_text_and_prompt['input_ids'],
              'start_embedding_text_and_prompt_attention': start_embedding_text_and_prompt['attention_mask'],
              'graph_embed_start_and_prompt_ids': graph_embed_start_and_prompt['input_ids'],
              'graph_embed_start_and_prompt_attention': graph_embed_start_and_prompt['attention_mask'],
              'landmarks_embed': landmarks_embed
              }

    return sample

  def __len__(self):
    return len(self.cellids)

  def calc_dist(self, start_point_cell, dist_matrix):

    label = self.cellid_to_label[start_point_cell]

    dists = dist_matrix[label]

    return dists