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
from typing import Any, Dict, Text, Tuple
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
  coord_to_cellid: Dict[str, int] = attr.ib()
  graph_embed_size: int = attr.ib()

  @classmethod
  def from_TextGeoSplit(cls, train, valid, test, unique_cellids,
                        unique_cellids_binary, label_to_cellid, coord_to_cellid, graph_embed_size):
    """Construct a TextGeoDataset."""
    return TextGeoDataset(
      train,
      valid,
      test,
      unique_cellids,
      unique_cellids_binary,
      label_to_cellid,
      coord_to_cellid,
      graph_embed_size,
    )

  @classmethod
  def load(cls, dataset_dir: Text, model_type: Text = None,
           s2_level: Text = None):
    if model_type:
      dataset_dir = os.path.join(dataset_dir, str(model_type))
    if s2_level:
      dataset_dir = os.path.join(dataset_dir, str(s2_level))

    train_path_dataset = os.path.join(dataset_dir, 'train.pth')
    valid_path_dataset = os.path.join(dataset_dir, 'valid.pth')
    test_path_dataset = os.path.join(dataset_dir, 'test.pth')
    unique_cellid_path = os.path.join(dataset_dir, "unique_cellid.npy")
    tensor_cellid_path = os.path.join(dataset_dir, "tensor_cellid.pth")
    label_to_cellid_path = os.path.join(dataset_dir, "label_to_cellid.npy")
    coord_to_cellid_path = os.path.join(dataset_dir, "coord_to_cellid.npy")
    graph_embed_size_path = os.path.join(dataset_dir, "graph_embed_size.npy")


    logging.info("Loading dataset from <== {}.".format(dataset_dir))
    train_dataset = torch.load(train_path_dataset)
    valid_dataset = torch.load(valid_path_dataset)
    test_dataset = torch.load(test_path_dataset)
    logging.info(f"Size of train set: {len(train_dataset)}" +
                 f", Size of validation set: {len(valid_dataset)}, Size of test set: {len(test_dataset)}")

    unique_cellid = np.load(unique_cellid_path, allow_pickle='TRUE')
    label_to_cellid = np.load(
      label_to_cellid_path, allow_pickle='TRUE').item()
    tens_cells = torch.load(tensor_cellid_path)
    coord_to_cellid = np.load(coord_to_cellid_path, allow_pickle='TRUE').item()
    graph_embed_size = np.load(graph_embed_size_path, allow_pickle='TRUE')
    logging.info(f"Loaded dataset with graph embedding size {graph_embed_size}")

    dataset_text = TextGeoDataset(
      train_dataset, valid_dataset, test_dataset,
      unique_cellid, tens_cells, label_to_cellid, coord_to_cellid, graph_embed_size)

    return dataset_text

  @classmethod
  def save(cls, dataset_text: Any, dataset_path: Text,
           graph_embed_size: int):
    os.mkdir(dataset_path)

    train_path_dataset = os.path.join(dataset_path, 'train.pth')
    valid_path_dataset = os.path.join(dataset_path, 'valid.pth')
    test_path_dataset = os.path.join(dataset_path, 'test.pth')
    unique_cellid_path = os.path.join(dataset_path, "unique_cellid.npy")
    tensor_cellid_path = os.path.join(dataset_path, "tensor_cellid.pth")
    label_to_cellid_path = os.path.join(dataset_path, "label_to_cellid.npy")
    coord_to_cellid_path = os.path.join(dataset_path, "coord_to_cellid.npy")
    graph_embed_size_path = os.path.join(dataset_path, "graph_embed_size.npy")

    torch.save(dataset_text.train, train_path_dataset)
    torch.save(dataset_text.valid, valid_path_dataset)
    torch.save(dataset_text.test, test_path_dataset)
    np.save(unique_cellid_path, dataset_text.unique_cellids)
    torch.save(dataset_text.unique_cellids_binary, tensor_cellid_path)
    np.save(label_to_cellid_path, dataset_text.label_to_cellid)
    np.save(coord_to_cellid_path, dataset_text.coord_to_cellid)
    np.save(graph_embed_size_path, graph_embed_size)

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
               model_type: str, dprob: util.DistanceProbability,
               cellid_to_coord, dist_matrix: pd.DataFrame,
               graph_embed_file:Any = None, is_dist: Boolean = False
               ):

    self.text_tokenizer = text_tokenizer
    self.s2_tokenizer = s2_tokenizer
    self.cellid_to_label = cellid_to_label
    self.cellid_to_coord = cellid_to_coord
    self.s2level = s2level
    self.is_dist = is_dist
    self.model_type = model_type
    self.graph_embed_file = graph_embed_file

    data = data.assign(end_point=data.end_point)

    data['cellid'] = data.end_point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))

    data['neighbor_cells'] = data.cellid.apply(
      lambda x: gutil.neighbor_cellid(x, unique_cells_df.cellid.tolist()))

    # Tokenize instructions.

    self.instruction_list = data.instructions.tolist()
    if 'T5' in model_type:
      # Add prompt
      data.instructions = [model_type + ": " + t for t in self.instruction_list]
      logging.info(data.instructions.iloc[0])

    self.encodings = self.text_tokenizer(
      data.instructions.tolist(), truncation=True,
      padding=True, add_special_tokens=True, max_length=200)

    data['far_cells'] = data.cellid.apply(
      lambda cellid: unique_cells_df[unique_cells_df['cellid'] == cellid].far.iloc[0])

    cellids_array = np.array(data.cellid.tolist())
    neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    far_cells_array = np.array(data.far_cells.tolist())

    self.end_point = data.end_point.apply(
      lambda x: gutil.tuple_from_point(x)).tolist()


    self.coords_end = data.cellid.apply(lambda x: cellid_to_coord[x]).tolist()

    self.labels = data.cellid.apply(lambda x: cellid_to_label[x]).tolist()

    self.start_cells = data.start_point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))

    self.coords_start = self.start_cells.apply(lambda x: cellid_to_coord[x]).tolist()

    self.start_point_labels = self.get_cell_to_lablel(self.start_cells.tolist())

    self.cellids = self.s2_tokenizer(cellids_array)


    self.neighbor_cells = self.s2_tokenizer(neighbor_cells_array)

    self.far_cells = self.s2_tokenizer(far_cells_array)


    self.start_text_input_list = [
      str(i).replace(':', f': Start at {str(s)}.') for s, i in zip(
        self.start_point_labels, data.instructions.tolist())]

    if graph_embed_file:
      self.graph_embed_end = data['cellid'].apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))
      self.graph_embed_start = self.start_cells.apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))

      data['landmarks_cells'] = data.landmarks.apply(
        lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

      self.graph_embed_landmarks = data.landmarks_cells.apply(
        lambda l: [util.get_valid_graph_embed(
          self.graph_embed_file, str(cell)) for cell in l])

      self.start_embed_text_input_list = [
        str(i).replace(':', f': Start at {str(s)}.') for s, i in zip(
          self.graph_embed_start, data.instructions.tolist())]

    else:
      self.graph_embed_start = np.zeros(len(self.cellids))

    self.landmarks_dist_raw = []

    data['landmarks_cells'] = data.landmarks.apply(
      lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

    self.landmark_label = self.get_cell_to_lablel(data['landmarks_cells'].tolist())

    for s_p, lan_l, lan_p in zip(data.start_point.tolist(), self.landmark_label, data.landmarks.tolist()):
      landmark_dist_cur = []
      for e_p, l in zip(lan_p, lan_l.split(";")):
        dist = round(gutil.get_distance_between_points(s_p, e_p))
        landmark_dist_cur.append(f"{l} distance: {dist}")

      self.landmarks_dist_raw.append('; '.join(landmark_dist_cur))

    if is_dist:
      logging.info(f"Calculating distances between {dist_matrix.shape[0]} cells")
      dist_lists = self.start_cells.apply(lambda start: self.calc_dist(start, dist_matrix))

      self.prob = dist_lists.mapply(
        lambda row: [dprob(dist) for dist in row.tolist()])

      self.prob = self.prob.tolist()


    self.set_generation_model(data)


    del self.graph_embed_file
    # del self.start_point_cells
    del self.s2_tokenizer
    del self.text_tokenizer

  def get_cell_to_lablel(self, list_cells):
    if isinstance(list_cells[0], list):
      labels = []
      for c_list in list_cells:
        list_lables = []
        for c in c_list:
          list_lables.append(self.cellid_to_coord[c])

        labels.append('; '.join(list_lables))

    else:
      labels = [
        util.get_valid_cell_label(self.cellid_to_coord, int(c)) for c in list_cells]

    return labels


  def set_generation_model(self, data):

    function_name = f"set_{self.model_type.replace('-','_')}"
    logging.info(f"Running function: {function_name}")
    set_data_func = getattr(self, function_name)
    input_text, output_text = set_data_func(data)

    self.print_sample(
      mode_expected=self.model_type,
      input=input_text[0],
      output=output_text[0])

    if 'T5' not in self.model_type:
      self.text_output_tokenized = output_text
      self.text_input_tokenized = input_text
      return
    self.text_output_tokenized = self.text_tokenizer(
      output_text, truncation=True, padding=True, add_special_tokens=True).input_ids

    self.text_input_tokenized = self.text_tokenizer(
      input_text, truncation=True, padding='max_length', add_special_tokens=True, max_length=200)

  def set_S2_Generation_T5_text_start_to_landmarks_dist(self, data):

    return self.start_text_input_list, self.landmarks_dist_raw


  def set_S2_Generation_T5_text_start_to_end_dist(self, data):

    self.end_dist_raw = [f"{e_l} distance: {round(gutil.get_distance_between_points(s_p, e_p))}"
                         for e_l, e_p, s_p in zip(
        self.coords_end, self.end_point, data.start_point)]

    return self.start_text_input_list, self.end_dist_raw


  def set_S2_Generation_T5_start_embedding_text_input(self, data):

    return self.start_text_input_list, self.coords_end

  def set_S2_Generation_T5_start_text_input(self, data):

    return self.start_text_input_list, self.coords_end


  def set_S2_Generation_T5_Landmarks(self, data):
    assert 'T5' in self.model_type and 'landmarks' in data, "Landmarks not processed"

    return data.instructions.tolist(), self.landmark_label


  def set_Landmarks_NER_2_S2_Generation_T5_Warmup(self, data):
    assert  ('T5' in self.model_type and 'landmarks_ner_and_point' in data)

    landmarks_ner_input = [
      f"{self.model_type}: {ner}" for ner, point in data.landmarks_ner_and_point.tolist()]

    data = data.assign(landmarks_ner_and_prompt_input=landmarks_ner_input)

    landmark_cells = [
      gutil.cellid_from_point(
        point, self.s2level) for ner, point in data.landmarks_ner_and_point.tolist()]

    landmark_label = self.get_cell_to_lablel(landmark_cells)

    return data.landmarks_ner_and_prompt_input.tolist(), landmark_label

  def set_Text_2_Landmarks_NER_Generation_T5_Warmup(self, data):
    assert 'landmarks_ner' in data, "Landmarks NER not processed"
    return data.instructions.tolist(), data.landmarks_ner.tolist()

  def set_S2_Generation_T5_Warmup_start_end_to_dist(self, data):

    dists_start_end = [
      f"Distance: {round(gutil.get_distance_between_points(s, e))}" for s, e in zip(
        self.end_point, data.start_point)]

    start_end_point_list_raw = [
      f"{self.model_type}: {str(e)}, {str(s)}" for s, e in zip(
        self.start_point_labels, self.coords_end)]

    return start_end_point_list_raw, dists_start_end


  def set_S2_Generation_T5_Warmup_start_end(self, data):

    assert 'T5' in self.model_type and 'route' in data, "Route not processed"

    start_end_point_list_raw = [
      f"{self.model_type}: {str(e)}, {str(s)}" for s, e in zip(
        self.start_point_labels, self.coords_end)]

    data['route_fixed'] = data.route_fixed.apply(
      lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

    route_fixed_label = self.get_cell_to_lablel(data.route_fixed.tolist())

    return start_end_point_list_raw, route_fixed_label

  def set_S2_Generation_T5(self, data):
    return data.instructions.tolist(), self.coords_end


  def set_S2_Generation_T5_Path(self, data):

    assert 'T5' in self.model_type and 'route' in data, "Route not processed"

    data['route'] = data.route.apply(
      lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

    route_label = self.get_cell_to_lablel(data.route.tolist())

    self.route = self.text_tokenizer(
      route_label, truncation=True, padding=True, add_special_tokens=True).input_ids

    return data.instructions.tolist(), route_label

  def set_S2_Generation_T5_text_start_embedding_to_landmarks_dist(self, data):
    return self.start_text_input_list, self.landmarks_dist_raw


  def set_S2_Generation_T5_text_start_embedding_to_landmarks(self, data):
    return self.start_text_input_list, self.landmark_label


  def set_S2_Generation_T5_Warmup_cell_embed_to_cell_label(self, data):

    return [self.model_type]*len(self.coords_start), self.coords_start

  def set_Classification_Bert(self, data):

    return self.encodings, self.cellids

  def set_Dual_Encoder_Bert(self, data):
    return self.encodings, self.cellids

  def print_sample(self, mode_expected, input, output):

    assert 'T5' not in mode_expected or mode_expected in input, \
      f"mode_expected: {mode_expected} \n input: {input}"

    if self.model_type == mode_expected:
      logging.info(
        f"\n Example {self.model_type}: \n" +
        f"  Input: '{input}'\n" +
        f"  Output: {output}\n" +
        f"  Goal: {self.coords_end[0]}\n" +
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

    cellid = torch.tensor(self.cellids[idx])

    neighbor_cells = torch.tensor(self.neighbor_cells[idx])
    far_cells = torch.tensor(self.far_cells[idx])
    end_point = torch.tensor(self.end_point[idx])

    label = torch.tensor(self.labels[idx])

    if self.is_dist:
      prob = torch.tensor(self.prob[idx])
    else:
      prob = torch.tensor([])

    if hasattr(self.text_input_tokenized, 'items'):
      text_input = {key: torch.tensor(val[idx])
                    for key, val in self.text_input_tokenized.items()}
    else:

      text_input = torch.tensor(self.text_input_tokenized[idx])

    graph_embed_start = self.graph_embed_start[idx]


    text_output = torch.tensor(self.text_output_tokenized[idx])


    sample = {'text': text_input, 'cellid': cellid,
              'neighbor_cells': neighbor_cells,
              'far_cells': far_cells, 'end_point': end_point, 'label': label,
              'prob': prob, 'text_output': text_output, 'graph_embed_start': graph_embed_start
              }

    return sample



  def __len__(self):
    return len(self.cellids)

  def calc_dist(self, start_point_cell, dist_matrix):

    label = self.cellid_to_label[start_point_cell]

    dists = dist_matrix[label]

    return dists

def coord_format(coord):
  x, y = coord
  return f'loc_{x} loc_{y}'
