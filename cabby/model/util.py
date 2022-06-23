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
'''Library to support models.'''

import scipy.stats

from typing import Tuple, Sequence, Optional, Dict, Text, Any

from absl import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn


from cabby.geo import util

CELLID_DIM = 64


class DistanceProbability:
  '''Calculates Gamma distribution probability of a given distance in meters.

  For details of the Gamma distribution see:
    https://en.wikipedia.org/wiki/Gamma_distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

  With the default shape of 2, shorter distances receive lower probability and 
  then peaks at 1 (unscaled) before dropping again. This allows the probability
  to be used to prefer values close to some scaled mean (given via the 
  `scale_meters` parameter.) Changing the shape will change this interpretation
  but could be used to make the values more spread out (higher value of 
  `gamma_shape`) or start high and drop (`gamma_shape` < 1). In the later case,
  this could be useful for situations where being extremely close is preferable.
  
  `scale_meters`: A factor (in meters) that gives the overall scale in meters
    that distance probabilities are required for. If `gamma_shape` is 2, then
    the distribution reaches its peak probability around `scale_meters`.

  `gamma_shape`: The shape of the Gamma distribution.

  '''
  def __init__(self, scale_meters: float, gamma_shape=2):
    assert(scale_meters > 0.0)
    self.scale_meters = scale_meters
    self.gamma_dist = scipy.stats.gamma(gamma_shape)

  def __call__(self, dist_meters: float) -> float:
    '''Computes the probability for a given distance in meters.'''
    assert(dist_meters >= 0.0), dist_meters
    return self.gamma_dist.pdf(dist_meters/self.scale_meters)

  def cdf(self, dist_meters: float) -> float:
    '''Computes the cumulative probability at a given distance in meters.'''
    assert(dist_meters >= 0.0)
    return self.gamma_dist.cdf(dist_meters/self.scale_meters)


def binary_representation(array_int: list, dim: int = CELLID_DIM
) -> np.ndarray:
  '''Converts an aray of integers to their binary vector with specific 
  dimension.
  Arguments:
    array_int: Array of integers to be converted to binary vectors.
    dim: The dimension of the vctor to output.
  Returns:
    An array of binary vectors.
  '''

  array_int = np.array(array_int)
  binary_range = 2**np.arange(dim, dtype=array_int.dtype)
  vector = array_int.reshape(-1,1)

  bin_rep = ((vector & binary_range) != 0).astype(int)
  return bin_rep


def save_checkpoint_trainer(save_path: Text, model:  torch.nn.Module, 
  valid_loss: float):
  '''Function for saving model.'''

  if save_path == None:
    return

  state_dict = {'model_state_dict': model.state_dict(),
          'valid_loss': valid_loss}

  torch.save(state_dict, save_path)
  logging.info(f'Model saved to ==> {save_path}')


def save_checkpoint(save_path: Text, model:  torch.nn.Module, 
  valid_loss: float):
  '''Function for saving model.'''

  if save_path == None:
    return

  if isinstance(model, nn.DataParallel):
    model_state_dict =  model.module.state_dict()
  else: 
    model_state_dict =  model.state_dict()
    
  state_dict = {'model_state_dict': model_state_dict,
          'valid_loss': valid_loss}
        
  torch.save(state_dict, save_path)
  logging.info(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path: Text, model:  torch.nn.Module,
  device: torch.device) -> Dict[Text, Sequence]:
  '''Function for loading model.'''

  if load_path == None:
    return

  state_dict = torch.load(load_path, map_location=device)

  if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(state_dict['model_state_dict'])

  else: 
    model.load_state_dict(state_dict['model_state_dict'])
  
  model.best_valid_loss = state_dict['valid_loss']
  logging.info(f'Model loaded from <== {load_path} with validation loss {model.best_valid_loss}')

  return state_dict


def save_metrics_last_only(save_path: Text,
         true_points_list: Sequence[Tuple[float, float]],
         pred_points_list: Sequence[Tuple[float, float]]):
  '''Function for saving results.'''

  if save_path == None:
    return
  state_dict = {
          'true_points_lat': [lat for lat, lon in true_points_list],
          'true_points_lon': [lon for lat, lon in true_points_list],
          'pred_points_lat': [lat for lat, lon in pred_points_list],
          'pred_points_lon': [lon for lat, lon in pred_points_list],
  }

  state_df = pd.DataFrame(state_dict)
  state_df.to_csv(save_path, sep = '\t', header=False)
  logging.info(f'Results saved to ==> {save_path}')


def save_metrics(save_path: Text,
         train_loss_list: Sequence[float],
         valid_loss_list: Sequence[float],
         global_steps_list: Sequence[int],
         valid_accuracy_list:  Sequence[float],
         true_points_list: Sequence[Sequence[Tuple[float, float]]],
         pred_points_list: Sequence[Sequence[Tuple[float, float]]]):
  '''Function for saving results.'''

  if save_path == None:
    return
  state_dict = {'train_loss_list': train_loss_list,
          'valid_loss_list': valid_loss_list,
          'global_steps_list': global_steps_list,
          'valid_accuracy_list': valid_accuracy_list,
          'true_points_list': true_points_list,
          'pred_points_list': pred_points_list}

  torch.save(state_dict, save_path)
  logging.info(f'Results saved to ==> {save_path}')


def load_metrics(load_path: Text, device: torch.device) -> Dict[Text, float]:
  '''Function for loading results.'''

  if load_path == None:
    return

  state_dict = torch.load(load_path, map_location=device)
  logging.info(f'Results loaded from <== {load_path}')

  return state_dict


def predictions_to_points(preds: Sequence,
  label_to_cellid: Dict[int, int]) -> Sequence[Tuple[float, float]]:
  default_cell = list(label_to_cellid.values())[0]
  cellids = [] 
  for label in preds:
    cellids.append(label_to_cellid[label] if label in label_to_cellid else default_cell)
  coords = util.get_center_from_s2cellids(cellids)
  return coords

def get_valid_label(dict_cells_lables: Dict, cellid: str):
  while cellid not in dict_cells_lables:
    cellid = util.neighbor_cellid(int(cellid))

  return dict_cells_lables[cellid]
  
