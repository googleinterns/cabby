
# Copyright 2020 The Flax Authors.
#
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


from typing import Tuple, Sequence, Optional, Dict, Text, Any

from absl import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn


from cabby.geo import util


def binary_representation(array_int: np.ndarray, dim: int
) -> np.ndarray:
  '''Converts an aray of integers to their binary vector with specific 
  dimension.
  Arguments:
    array_int: Array of integers to be converted to binary vectors.
    dim: The dimension of the vctor to output.
  Returns:
    An array of binary vectors.
  '''

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
  logging.info(f'Model loaded from <== {load_path}')

  if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(state_dict['model_state_dict'])

  else: 
    model.load_state_dict(state_dict['model_state_dict'])
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
  cellids = [label_to_cellid[label] for label in preds]
  coords = util.get_center_from_s2cellids(cellids)
  return coords
