
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
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader

from cabby.model.text import util


def evaluate(model: torch.nn.Module,
       valid_loader: DataLoader,
       device: torch.device,
       label_to_cellid: Dict
       ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  '''Validate the modal.'''

  model.eval()

  loss_val_total = 0

  # Validation loop.
  true_points_list, pred_points_list, predictions, true_vals = [], [], [], []
  for batch, points in valid_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(
      input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss_val_total += loss.mean().item()
    label_ids = labels.cpu().numpy()
    true_vals.append(label_ids)
    logits = outputs.logits.detach().cpu().numpy()
    predictions.append(logits)
    points = points.cpu().numpy()
    true_points_list.append(points)
    pred_points = util.predictions_to_points(logits, label_to_cellid)
    pred_points_list.append(pred_points)

  # Evaluation.
  pred_points_list = np.concatenate(pred_points_list, axis=0)
  true_points_list = np.concatenate(true_points_list, axis=0)
  predictions = np.concatenate(predictions, axis=0)
  true_vals = np.concatenate(true_vals, axis=0)
  average_valid_loss = loss_val_total / len(valid_loader)

  return (average_valid_loss, predictions, true_vals, true_points_list,
      pred_points_list)


def accuracy_cells(labels: np.ndarray, preds: np.ndarray) -> float:
  '''Accuracy classification score.'''
  preds_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  return accuracy_score(labels_flat, preds_flat)


def train_model(model:  torch.nn.Module,
        device: torch.device,
        optimizer: AdamW,
        file_path: Text,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        label_to_cellid: Dict[int, int],
        num_epochs: int,
        best_valid_loss: float = float("Inf")):
  '''Main funcion for training model.'''
  # initialize running values
  running_loss = 0.0
  global_step = 0
  valid_accuracy_list, train_loss_list, valid_loss_list = [], [], []
  global_steps_list, true_points_list, pred_points_list = [], [], []

  # Training loop.
  model.train()

  for epoch in range(num_epochs):
    for batch, _ in train_loader:
      optimizer.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(
        input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss
      loss.mean().backward()
      optimizer.step()

      # Update running values.
      running_loss += loss.mean().item()
      global_step += 1

    # Evaluation step.
    valid_loss, predictions, true_vals, true_points, pred_points = evaluate(
      model, valid_loader, device, label_to_cellid)

    average_train_loss = running_loss / labels.shape[0]
    accuracy = accuracy_cells(true_vals, predictions)
    train_loss_list.append(average_train_loss)
    valid_loss_list.append(valid_loss)
    global_steps_list.append(global_step)
    valid_accuracy_list.append(accuracy)
    true_points_list.append(true_points)
    pred_points_list.append(pred_points)

    # Resetting running values.
    running_loss = 0.0

    logging.info('Epoch [{}/{}], Step [{}/{}], \
        Accuracy: {:.4f},Train Loss: {:.4f}, Valid Loss: {:.4f}'
           .format(epoch+1, num_epochs, global_step,
               num_epochs*len(train_loader), accuracy,
               average_train_loss, valid_loss))

    # Save model and results in checkpoint.
    if best_valid_loss > valid_loss:
      best_valid_loss = valid_loss
      util.save_checkpoint(file_path + '/' + 'model.pt',
              model, best_valid_loss)
      util.save_metrics(file_path + '/' + 'metrics.pt', train_loss_list,
             valid_loss_list, global_steps_list, valid_accuracy_list, true_points_list, pred_points_list)

      model.train()

  logging.info('Finished Training.')
