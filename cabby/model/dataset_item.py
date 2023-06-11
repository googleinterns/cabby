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
from pyproj import Geod
from geopandas import GeoDataFrame, GeoSeries
import numpy as np
import os
import pandas as pd
import re
from shapely.geometry.point import Point
from shapely.geometry import box, mapping, LineString
import sys
import swifter
from s2geometry import pywraps2 as s2
from typing import Any, Dict, Text, Tuple, List
import torch
from transformers import DistilBertTokenizerFast, T5Tokenizer

import mapply

import attr

from cabby.geo import regions
from cabby.geo import util as gutil
from cabby.model import util

os.environ["TOKENIZERS_PARALLELISM"] = "false"

mapply.init(
  n_workers=-1,
  progressbar=True,
)
T5_TYPE = "t5-small"
BERT_TYPE = 'distilbert-base-uncased'

geod = Geod(ellps="WGS84")


MODELS = [
  'Dual-Encoder-Bert',
  'Classification-Bert',
  'S2-Generation-T5',
  'S2-Generation-T5-Landmarks',
  'S2-Generation-T5-Warmup-start-end',
  'Text-2-Landmarks-NER-Generation-T5-Warmup',
  'Landmarks-NER-2-S2-Generation-T5-Warmup',
  'S2-Generation-T5-Path',
  'S2-Generation-T5-start-text-input',
  'S2-Generation-T5-Warmup-cell-embed-to-cell-label',
  'S2-Generation-T5-start-embedding-text-input',
  'S2-Generation-T5-Warmup-start-end-to-dist',
  'S2-Generation-T5-text-start-to-end-dist',
  'S2-Generation-T5-text-start-to-landmarks-dist',
  'S2-Generation-T5-text-start-embedding-to-landmarks',
  'S2-Generation-T5-text-start-embedding-to-landmarks-dist'
]


tokenizerT5 = T5Tokenizer.from_pretrained(T5_TYPE)



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
  train_set: Any = attr.ib()
  dev_set: Any = attr.ib()
  test_set: Any = attr.ib()

  @classmethod
  def from_TextGeoSplit(cls, train_set, dev_set, test_set):
    """Construct a TextGeoDataset."""
    return TextGeoDataset(
      train_set,
      dev_set,
      test_set
    )

  @classmethod
  def load_set(cls, dataset_dir, set_region):

    logging.info(f"Loading {set_region}-set from <== {dataset_dir}.")

    path_dataset = os.path.join(dataset_dir, f'{set_region.lower()}.pth')
    data_set = torch.load(path_dataset)
    logging.info(f"Size of {set_region}-set: {len(data_set)}")

    logging.info(f"Number of cells in the {set_region} region: {len(data_set.cellid_to_label)}")

    active_region = regions.get_region(set_region)
    area = abs(geod.geometry_area_perimeter(active_region.polygon)[0])
    logging.info(f'{set_region} region area: {round(area, 3)} m^2')


    loc_x_loc_y_list = [xy.split(' ') for xy in list(data_set.cellid_to_coord.values())]

    max_x = max([int(re.search(r'\d+', xy[0]).group()) for xy in loc_x_loc_y_list])
    max_y = max([int(re.search(r'\d+', xy[1]).group()) for xy in loc_x_loc_y_list])

    logging.info(f"Grid axis in {set_region}: X: 0-{max_x} / Y: 0-{max_y}")

    return data_set


  @classmethod
  def load(cls, dataset_dir, model_type, s2_level, train_region, dev_region, test_region):
    if model_type:
      dataset_dir = os.path.join(dataset_dir, str(model_type))
    if s2_level:
      dataset_dir = os.path.join(dataset_dir, str(s2_level))
    
    train_set = cls.load_set(dataset_dir, train_region)
    
    dev_set = cls.load_set(dataset_dir, dev_region)
   
    test_set = cls.load_set(dataset_dir, test_region)
    
    return  cls.from_TextGeoSplit(train_set, dev_set, test_set)


  @classmethod
  def save(cls, dataset_text: Any, dataset_dir: Text):
    os.mkdir(dataset_dir)

    train_path_dataset = os.path.join(dataset_dir, f'{dataset_text.train_set.region.lower()}.pth')
    valid_path_dataset = os.path.join(dataset_dir, f'{dataset_text.dev_set.region.lower()}.pth')
    test_path_dataset = os.path.join(dataset_dir, f'{dataset_text.test_set.region.lower()}.pth')


    torch.save(dataset_text.train_set, train_path_dataset)
    torch.save(dataset_text.dev_set, valid_path_dataset)
    torch.save(dataset_text.test_set, test_path_dataset)

    logging.info("Saved data to ==> {}.".format(dataset_dir))

class TextGeoSplit(torch.utils.data.Dataset):
  """A split of of the dataset.
  `points`: The ground true end-points of the samples.
  `labels`: The ground true label of the cellid.
  `cellids`: The ground truth S2Cell id.
  `neighbor_cells`: One neighbor cell id of the ground truth S2Cell id.
  `far_cells`: One far away cell id (in the region defined) of the ground truth
  'dprob': Gamma distribution probability.
  S2Cell id.
  """

  def __init__(self, region: str, data: pd.DataFrame, s2level: int,
               cellid_to_label: Dict[int, int],
               unique_cellids: List[int],
               model_type: str, dprob: util.DistanceProbability,
               far_cell_dist: int,
               set_type: str,
               graph_embed_size: int,
               graph_embed_file:Any = None, is_dist: Boolean = False
               ):

    logging.info(f"Creating dataset for {region}")

    self.region = region
    self.cellid_to_label = cellid_to_label
    self.s2level = s2level
    self.is_dist = is_dist
    self.model_type = model_type
    self.graph_embed_file = graph_embed_file
    self.set_type = set_type
    self.unique_cellids = unique_cellids
    self.graph_embed_size = graph_embed_size
    
    self.label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellids)}

    points = gutil.get_centers_from_s2cellids(self.unique_cellids)

    unique_cells_df = pd.DataFrame(
      {'point': points, 'cellid': self.unique_cellids})

    self.cellid_to_coord, self.coord_to_cellid = self.create_grid(unique_cells_df)

    self.set_tokenizers()

    vec_cells = self.s2_tokenizer(unique_cellids)
    self.unique_cellids_binary = torch.tensor(vec_cells)


    logging.info(f"Created grid")
    dist_matrix = unique_cells_df.point.mapply(
      lambda x: calc_dist(x, unique_cells_df)
    )
    dist_matrix = dist_matrix.to_numpy()

    unique_cells_df['far'] = unique_cells_df.point.swifter.apply(
      lambda x: gutil.far_cellid(x, unique_cells_df, far_cell_dist))
   

    data = data.assign(end_point=data.end_point)

    data['cellid'] = data.end_point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))


    # Tokenize instructions.

    self.instruction_list = data.instructions.tolist()
    if 'T5' in model_type:
      # Add prompt
      data.instructions = [model_type + ": " + t for t in self.instruction_list]
      logging.info(data.instructions.iloc[0])

    self.encodings = self.text_tokenizer(
      data.instructions.tolist(), truncation=True,
      padding=True, add_special_tokens=True, max_length=200)

    cellids_array = np.array(data.cellid.tolist())

    self.end_point = data.end_point.apply(
      lambda x: gutil.tuple_from_point(x)).tolist()

    self.start_point = data.start_point.apply(
      lambda x: gutil.tuple_from_point(x)).tolist()

    self.coords_end = data.cellid.apply(lambda x: util.get_valid_cell_label(self.cellid_to_coord, x)).tolist()

    self.labels = data.cellid.apply(lambda x: util.get_valid_cell_label(cellid_to_label, x)).tolist()

    self.start_cells = data.start_point.apply(
      lambda x: gutil.cellid_from_point(x, s2level))

    self.start_cells_list = self.start_cells.tolist()

    self.coords_start = self.start_cells.apply(lambda x: util.get_valid_cell_label(self.cellid_to_coord, x)).tolist()

    self.start_point_labels = self.get_cell_to_lablel(self.start_cells.tolist())

    self.cellids = self.s2_tokenizer(cellids_array)

    if 'Dual-Encoder-Bert' in model_type:
      data['neighbor_cells'] = data.cellid.apply(
        lambda x: gutil.neighbor_cellid(x, unique_cells_df.cellid.tolist()))
      data['far_cells'] = data.cellid.apply(
        lambda cellid: unique_cells_df[unique_cells_df['cellid'] == cellid].far.iloc[0])
      neighbor_cells_array = np.array(data.neighbor_cells.tolist())
      far_cells_array = np.array(data.far_cells.tolist())
      self.neighbor_cells = self.s2_tokenizer(neighbor_cells_array)

      self.far_cells = self.s2_tokenizer(far_cells_array)
    else:
      self.neighbor_cells = [0]*len(self.cellids)
      self.far_cells = [0]*len(self.cellids)


    self.start_text_input_list = [
      str(i).replace(':', f': Start at {str(s)}.') for s, i in zip(
        self.start_point_labels, data.instructions.tolist())]

    if graph_embed_file:
      self.graph_embed_end = data['cellid'].apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))
      self.graph_embed_start = self.start_cells.apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))

      if 'landmarks' in data:
        data['landmarks_cells'] = data.landmarks.apply(
          lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

        self.graph_embed_landmarks = data.landmarks_cells.apply(
          lambda l: [util.get_valid_graph_embed(
            self.graph_embed_file, str(cell)) for cell in l])
      else:
        self.graph_embed_landmarks = ['0']*data.instructions.shape[0]

      self.start_embed_text_input_list = [
        str(i).replace(':', f': Start at {str(s)}.') for s, i in zip(
          self.graph_embed_start, data.instructions.tolist())]

    else:
      self.graph_embed_start = np.zeros(len(self.cellids))

    self.landmarks_dist_raw = []

    if 'landmarks' in data:
      data['landmarks_cells'] = data.landmarks.apply(
        lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

      self.landmark_label = self.get_cell_to_lablel(data['landmarks_cells'].tolist())

      for s_p, lan_l, lan_p in zip(data.start_point.tolist(), self.landmark_label, data.landmarks.tolist()):
        landmark_dist_cur = []
        for e_p, l in zip(lan_p, lan_l.split(";")):
          dist = round(gutil.get_distance_between_points(s_p, e_p))
          landmark_dist_cur.append(f"{l}; distance: {dist}")

        self.landmarks_dist_raw.append('; '.join(landmark_dist_cur))

    else:
      logging.info(self.region)
      self.landmark_label = ['0']*data.shape[0]
    if is_dist:
      logging.info(f"Calculating distances between {dist_matrix.shape[0]} cells")
      dist_lists = self.start_cells.apply(lambda start: self.calc_dist(start, dist_matrix))

      self.prob = dist_lists.mapply(
        lambda row: [dprob(dist) for dist in row.tolist()])

      self.prob = self.prob.tolist()


    self.set_generation_model(data)
    self.data = data




  def set_tokenizers(self):
    assert self.model_type in MODELS
    if self.model_type in ['Dual-Encoder-Bert', 'Classification-Bert']:
      self.text_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_TYPE)
      self.s2_tokenizer = util.binary_representation
    elif 'T5' in self.model_type:
      self.text_tokenizer = tokenizerT5
      self.s2_tokenizer = self.tokenize_cell

  def tokenize_cell(self, list_cells):
    if isinstance(list_cells[0], list):
      labels = []
      for c_list in list_cells:
        list_lables = []
        for c in c_list:
          list_lables.append(self.cellid_to_coord[c])

        labels.append('; '.join(list_lables))

    else:
      labels = [util.get_valid_cell_label(self.cellid_to_coord, c) for c in list_cells]

    return tokenizerT5(
      labels, padding=True, truncation=True).input_ids

  def create_grid(self, unique_cells_df):

    unique_cells_df['lon'] = unique_cells_df.point.apply(lambda p: p.x)
    unique_cells_df['lat'] = unique_cells_df.point.apply(lambda p: p.y)

    unique_cells_df = unique_cells_df.sort_values(by=['lat', 'lon'], ascending=True)

    cellid_to_coord = {}
    coord_to_cellid = {}

    full_cellid_list = unique_cells_df.cellid.tolist()
    empty_cellid_list = []

    x_current = 0
    y_current = 0

    current_cellid = unique_cells_df.cellid.tolist()[0]
    size_region = len(unique_cells_df)
    min_cell = current_cellid
    tmp = []
    begin_cell = None

    for down_idx in range(20):
      cell = s2.S2CellId(current_cellid)
      down_cell = cell.GetEdgeNeighbors()[1]
      current_cellid = down_cell.id()

    for up_idx in range(20):
      cell = s2.S2CellId(current_cellid)
      up_cell = cell.GetEdgeNeighbors()[3]
      current_cellid = up_cell.id()

      for left_idx in range(round(size_region/20)):
        cell = s2.S2CellId(current_cellid)
        left_cell = cell.GetEdgeNeighbors()[0]
        left_cell_id = left_cell.id()
        current_cellid = left_cell_id
        if current_cellid in full_cellid_list:
          begin_cell = current_cellid

      if begin_cell:
        break

      for right_idx in range(round(size_region/20)):
        cell = s2.S2CellId(current_cellid)
        right_cell = cell.GetEdgeNeighbors()[2]
        right_cell_id = right_cell.id()
        current_cellid = right_cell_id
        if current_cellid in full_cellid_list:
          begin_cell = current_cellid
          break

      if begin_cell:
        break

    current_cellid = begin_cell

    status_cell_list = (len(full_cellid_list), 0)

    while True:
      cell = s2.S2CellId(current_cellid)
      next_cell = cell.GetEdgeNeighbors()[2]
      next_cellid = next_cell.id()


      is_next = False

      if current_cellid in full_cellid_list:
        cellid_to_coord[current_cellid] = (x_current, y_current)

        full_cellid_list.remove(current_cellid)
        empty_cellid_list.append(current_cellid)

        status_cell_list = (len(full_cellid_list), 0)
        x_current += 1
        current_cellid = next_cellid
        is_next = True

      if not is_next:
        y_current += 1

        upper_cell = cell.GetEdgeNeighbors()[3]
        current_cellid = upper_cell.id()
        prev_cell = upper_cell.GetEdgeNeighbors()[0]
        prev_cell_id = prev_cell.id()

        counter_move_left = 0
        while prev_cell_id in full_cellid_list and counter_move_left<size_region:
          counter_move_left += 1
          cell = s2.S2CellId(current_cellid)
          prev_cell = cell.GetEdgeNeighbors()[0]
          prev_cell_id = prev_cell.id()
          current_cellid = prev_cell_id
          x_current -= 1

        for i in range(round(size_region/10)):
          cell = s2.S2CellId(current_cellid)
          prev_cell = cell.GetEdgeNeighbors()[0]
          prev_cell_id = prev_cell.id()
          current_cellid = prev_cell_id
          x_current -= 1

        counter_move_right = 0
        while current_cellid not in full_cellid_list and counter_move_right<size_region:
          counter_move_right += 1
          cell = s2.S2CellId(current_cellid)
          next_cell = cell.GetEdgeNeighbors()[2]
          next_cell_id = next_cell.id()
          current_cellid = next_cell_id
          x_current += 1

      size_cell_list = len(full_cellid_list)
      if size_cell_list == 0:
        break

      if size_cell_list==status_cell_list[0]:
        status_cell_list = (size_cell_list, status_cell_list[1]+1)

      if status_cell_list[1]>100:
        sys.exit(f"Problem with creating grid. " +
                 f"There are still {size_cell_list} cells not in grid: {full_cellid_list}. " +
                 f"The beginig cell: {begin_cell}")

    min_x = min(cellid_to_coord.items(), key=lambda x: x[1][0])[1][0]

    if min_x < 0:

      add_x = -1 * min_x
      new_cellid_to_coord = {}
      for cellid, (x, y) in cellid_to_coord.items():

        new_x = x + add_x
        new_cellid_to_coord[cellid] = (new_x, y)

      cellid_to_coord = new_cellid_to_coord

    min_x = min(cellid_to_coord.items(), key=lambda x: x[1][0])[1][0]
    min_y = min(cellid_to_coord.items(), key=lambda x: x[1][1])[1][1]

    assert min_x >= 0 and min_y >= 0

    max_x = max(cellid_to_coord.items(), key=lambda x: x[1][0])[1][0]
    max_y = max(cellid_to_coord.items(), key=lambda x: x[1][1])[1][1]

    logging.info(f"Grid axis: X: 0-{max_x} / Y: 0-{max_y}")

    coord_format_to_cellid = {coord_format(coord): cellid for cellid, coord in cellid_to_coord.items()}
    cellid_to_coord_format = {cellid: coord_format(coord) for cellid, coord in cellid_to_coord.items()}

    assert len(
      full_cellid_list) == 0, f"full_cellid_list: {len(full_cellid_list)} empty_cellid_list:{len(empty_cellid_list)}"
    assert len(cellid_to_coord_format) == unique_cells_df.cellid.shape[0]

    return cellid_to_coord_format, coord_format_to_cellid

  def save(self, dataset_dir):
    
    os.makedirs(dataset_dir, exist_ok=True)

    path_dataset = os.path.join(dataset_dir, f'{self.set_type}.pth')
    unique_cellid_path = os.path.join(dataset_dir, f"{self.set_type}_unique_cellid.npy")
    tensor_cellid_path = os.path.join(dataset_dir, f"{self.set_type}_tensor_cellid.pth")
    label_to_cellid_path = os.path.join(dataset_dir, f"{self.set_type}_label_to_cellid.npy")
    coord_to_cellid_path = os.path.join(dataset_dir, f"{self.set_type}_coord_to_cellid.npy")
    graph_embed_size_path = os.path.join(dataset_dir, f"{self.set_type}_graph_embed_size.npy")

    torch.save(self.data, path_dataset)
    np.save(unique_cellid_path, self.unique_cellids)
    torch.save(self.unique_cellids_binary, tensor_cellid_path)
    np.save(label_to_cellid_path, self.label_to_cellid)
    np.save(coord_to_cellid_path, self.coord_to_cellid)
    np.save(graph_embed_size_path, self.graph_embed_size)

    logging.info(f"Saved {self.set_type} data to ==> {dataset_dir}.")


  def get_cell_to_lablel(self, list_cells):
    if isinstance(list_cells[0], list):
      labels = []
      for c_list in list_cells:
        list_lables = []
        for c in c_list:
          list_lables.append(util.get_valid_cell_label(self.cellid_to_coord, c))

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
    start_point = torch.tensor(self.start_point[idx])

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
              'neighbor_cells': neighbor_cells, 'far_cells': far_cells, 
              'end_point': end_point, 'start_point': start_point, 'label': label, 'prob': prob, 
              'text_output': text_output, 'graph_embed_start': graph_embed_start,
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


def calc_dist(start, unique_cells_df):
  dists = unique_cells_df.apply(
    lambda end: gutil.get_distance_between_points(start, end.point), axis=1)

  return dists