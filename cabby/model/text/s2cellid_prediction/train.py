
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
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader


def save_checkpoint(save_path: Text, model:  torch.nn.Module, valid_loss: int):
  '''Funcion for saving model.'''

  if save_path == None:
    return

  state_dict = {'model_state_dict': model.state_dict(),
          'valid_loss': valid_loss}

  torch.save(state_dict, save_path)
  logging.info(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path: Text, model:  torch.nn.Module,
          device: torch.device) -> Dict[Text, Sequence]:
  '''Funcion for loading model.'''

  if load_path == None:
    return

  state_dict = torch.load(load_path, map_location=device)
  logging.info(f'Model loaded from <== {load_path}')

  model.load_state_dict(state_dict['model_state_dict'])
  return state_dict


def save_metrics(save_path: Text,
         train_loss_list: Sequence[float],
         valid_loss_list: Sequence[float],
         global_steps_list: Sequence[int],
         valid_accuracy_list:  Sequence[float]):
  '''Funcion for saving results.'''

  if save_path == None:
    return

  state_dict = {'train_loss_list': train_loss_list,
          'valid_loss_list': valid_loss_list,
          'global_steps_list': global_steps_list,
          'valid_accuracy_list': valid_accuracy_list}

  torch.save(state_dict, save_path)
  logging.info(f'Results saved to ==> {save_path}')


def load_metrics(load_path: Text, device: torch.device) -> Dict[Text, float]:
  '''Funcion for loading results.'''

  if load_path == None:
    return

  state_dict = torch.load(load_path, map_location=device)
  logging.info(f'Results loaded from <== {load_path}')

  return state_dict


def train_model(model:  torch.nn.Module,
        device: torch.device,
        optimizer: AdamW,
        file_path: Text,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        eval_every: int,
        num_epochs: int,
        best_valid_loss: float = float("Inf")):
  '''Main funcion for training model.'''
  # initialize running values
  running_loss = 0.0
  valid_running_loss = 0.0
  global_step = 0
  valid_accuracy_list = []
  train_loss_list = []
  valid_loss_list = []
  global_steps_list = []

  # Training loop.
  model.train()

  for epoch in range(num_epochs):
    for batch in train_loader:
      optimizer.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(
        input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      loss.backward()
      optimizer.step()

      # Update running values.
      running_loss += loss.item()
      global_step += 1

      # Evaluation step.
      if global_step % eval_every == 0:
        model.eval()
        total, correct = 0, 0

        # Validation loop.
        for batch in valid_loader:
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(
            input_ids, attention_mask=attention_mask, labels=labels)
          loss = outputs.loss
          logits = outputs.logits
          valid_running_loss += loss.item()
          topv, topi = logits.squeeze().topk(1)
          topi = topi.squeeze().detach().cpu().numpy()

          label_ids = labels.to('cpu').numpy()
          correct += np.sum(label_ids == topi)
          total += label_ids.shape[0]

        # Evaluation.
        average_train_loss = running_loss / eval_every
        average_valid_loss = valid_running_loss / len(valid_loader)
        accuracy = 100*correct/total
        train_loss_list.append(average_train_loss)
        valid_loss_list.append(average_valid_loss)
        global_steps_list.append(global_step)
        valid_accuracy_list.append(accuracy)

        # Resetting running values.
        running_loss = 0.0
        valid_running_loss = 0.0
        model.train()

        logging.info('Epoch [{}/{}], Step [{}/{}], \
        Accuracy: {:.4f},Train Loss: {:.4f}, Valid Loss: {:.4f}'
               .format(epoch+1, num_epochs, global_step,
                   num_epochs*len(train_loader), accuracy,
                   average_train_loss, average_valid_loss))

        # Save model and results in checkpoint.
        if best_valid_loss > average_valid_loss:
          best_valid_loss = average_valid_loss
          save_checkpoint(file_path + '/' + 'model.pt',
                  model, best_valid_loss)
          save_metrics(file_path + '/' + 'metrics.pt', train_loss_list,
                 valid_loss_list, global_steps_list, valid_accuracy_list)

  save_metrics(file_path + '/' + 'metrics.pt', train_loss_list,
         valid_loss_list, global_steps_list, valid_accuracy_list)
  logging.info('Finished Training.')
