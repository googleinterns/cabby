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


from typing import Dict, Sequence, Any

from absl import logging
import numpy as np
import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader

from cabby.evals import utils as eu
from cabby.model import util


class Trainer:
  def __init__(
    self,
    model: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Adam,
    file_path: str,
    train_loader: Any,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    unique_cells: Sequence[int],
    num_epochs: int,
    cells_tensor: torch.tensor,
    label_to_cellid: Dict[int, int],
    is_single_sample_train: bool = False,
    best_valid_loss: float = float("Inf")

  ):

    self.model = model
    self.device = device
    self.optimizer = optimizer
    self.file_path = file_path
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.test_loader = test_loader
    self.unique_cells = unique_cells
    self.num_epochs = num_epochs
    self.cells_tensor = cells_tensor.float().to(self.device)
    self.label_to_cellid = label_to_cellid
    self.cos = nn.CosineSimilarity(dim=2)
    self.best_valid_loss = best_valid_loss
    if not os.path.exists(self.file_path):
      os.mkdir(self.file_path)
    self.model_path = os.path.join(self.file_path, 'model.pt')
    self.metrics_path = os.path.join(self.file_path, 'metrics.tsv')
    self.is_single_sample_train = is_single_sample_train
    self.evaluator = eu.Evaluator()

  def evaluate(self, validation_set: bool = True):
    '''Validate the model.'''

    if validation_set:
      data_loader = self.valid_loader
    else:
      data_loader = self.test_loader
      logging.info("Starting evaluation.")

    # Validation loop.
    true_points_list, pred_points_list = [], []
    predictions_list, true_vals = [], []
    correct = 0
    total = 0
    loss_val_total = 0

    self.model.eval()
    with torch.no_grad():
      for batch in data_loader:
        text = {key: val.to(self.device) for key, val in batch['text'].items()}
        cellids = batch['cellid'].float().to(self.device)

        batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

        loss = self.model(
          text,
          cellids,
          batch
        )

        loss_val_total += loss

        predictions = self.model.predict(
          text, self.cells_tensor, self.label_to_cellid, batch)

        predictions_list.append(predictions)
        labels = batch['label'].cpu()
        true_vals.append(labels)
        true_points_list.append(batch['end_point'].cpu())

    true_points_list = np.concatenate(true_points_list, axis=0)
    pred_points_list = np.concatenate(predictions_list, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    average_valid_loss = loss_val_total / len(data_loader)

    return (average_valid_loss, predictions_list, true_vals,
            true_points_list, pred_points_list)

  def train_model(self):

    '''Main function for training model.'''
    # Initialize running values.
    global_step = 0

    # Training loop.
    self.model.train()

    for epoch in range(self.num_epochs):
      running_loss = 0.0
      logging.info("Epoch number: {}".format(epoch))
      for batch_idx, batch in enumerate(self.train_loader):
        self.optimizer.zero_grad()
        text = {key: val.to(self.device) for key, val in batch['text'].items()}
        cellids = batch['cellid'].float().to(self.device)
        batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

        loss = self.model(
          text,
          cellids,
          batch
        )

        loss.backward()

        self.optimizer.step()

        # Update running values.
        running_loss += loss.item()
        global_step += 1

        if self.is_single_sample_train:
          break

      # Evaluation step.
      valid_loss, predictions, true_vals, true_points, pred_points = self.evaluate()

      average_train_loss = running_loss / (batch_idx + 1)

      # Resetting running values.
      running_loss = 0.0

      logging.info('Epoch [{}/{}], Step [{}/{}], \
          Train Loss: {:.4f}, Valid Loss: {:.4f}'
                   .format(epoch + 1, self.num_epochs, global_step,
                           self.num_epochs * len(self.train_loader),
                           average_train_loss, valid_loss))

      # Save model and results in checkpoint.
      if self.best_valid_loss > valid_loss:
        self.best_valid_loss = valid_loss
        util.save_checkpoint(self.model_path, self.model, self.best_valid_loss)

      self.model.train()

      if self.is_single_sample_train:
        return

    logging.info('Finished Training.')

    model_state = util.load_checkpoint(self.model_path, self.model, self.device)
    valid_loss = model_state['valid_loss']

    logging.info(
      f'Loaded best model (with validation loss {valid_loss}) for testing.')

    logging.info('Start testing.')

    test_loss, predictions, true_vals, true_points, pred_points = self.evaluate(
      validation_set=False)

    util.save_metrics_last_only(
      self.metrics_path,
      true_points,
      pred_points)

    evaluator = eu.Evaluator()
    error_distances = evaluator.get_error_distances(self.metrics_path)
    evaluator.compute_metrics(error_distances)

    if not self.model.is_generation:
      self.save_cell_embed()

  def multi_train_model(self, model_types):

    '''Main function for training model.'''
    # Initialize running values.
    global_step = 0

    # Training loop.
    self.model.train()

    for epoch in range(self.num_epochs):
      running_loss = 0.0
      logging.info("Epoch number: {}".format(epoch))
      for batches in zip(*self.train_loader):

        loss = torch.tensor([0.0]).to(self.device)
        self.optimizer.zero_grad()
        for batch_idx, model_type in zip(range(len(self.train_loader)),model_types):
          self.model_type = model_type
          batch = batches[batch_idx]
          text = {key: val.to(self.device) for key, val in batch['text'].items()}
          cellids = batch['cellid'].float().to(self.device)
          batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

          loss += self.model(
            text,
            cellids,
            batch
          )

        loss.backward()

        self.optimizer.step()

        # Update running values.
        running_loss += loss.item() / len(self.train_loader)
        global_step += 1

      # Evaluation step.
      valid_loss, predictions, true_vals, true_points, pred_points = self.evaluate()

      average_train_loss = running_loss / (batch_idx + 1)

      # Resetting running values.
      running_loss = 0.0

      logging.info('Epoch [{}/{}], Step [{}/{}], \
          Train Loss: {:.4f}, Valid Loss: {:.4f}'
                   .format(epoch + 1, self.num_epochs, global_step,
                           self.num_epochs * len(self.train_loader),
                           average_train_loss, valid_loss))

      # Save model and results in checkpoint.
      if self.best_valid_loss > valid_loss:
        self.best_valid_loss = valid_loss
        util.save_checkpoint(self.model_path, self.model, self.best_valid_loss)

      self.model.train()

      if self.is_single_sample_train:
        return

    logging.info('Finished Training.')

    model_state = util.load_checkpoint(self.model_path, self.model, self.device)
    valid_loss = model_state['valid_loss']

    logging.info(
      f'Loaded best model (with validation loss {valid_loss}) for testing.')

    logging.info('Start testing.')

    test_loss, predictions, true_vals, true_points, pred_points = self.evaluate(
      validation_set=False)

    util.save_metrics_last_only(
      self.metrics_path,
      true_points,
      pred_points)

    evaluator = eu.Evaluator()
    error_distances = evaluator.get_error_distances(self.metrics_path)
    evaluator.compute_metrics(error_distances)

    if not self.model.is_generation:
      self.save_cell_embed()

  def save_cell_embed(self):
    if isinstance(self.model, nn.DataParallel):
      cell_embed = self.model.module.cellid_main(self.cells_tensor)
    else:
      cell_embed = self.model.cellid_main(self.cells_tensor)
    cellid_to_embed = {
      cell: embed for cell, embed in zip(self.unique_cells, cell_embed)}
    path_to_save = os.path.join(self.file_path, 'cellid_to_embedding.pt')
    torch.save(cellid_to_embed, path_to_save)
    logging.info(f'Cell embedding saved to ==> {path_to_save}')


def infer_text(model: torch.nn.Module, text: str):
  if isinstance(model, nn.DataParallel):
    return model.module.text_embed(text)
  else:
    return model.text_embed(text)