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


from typing import Dict, Sequence 

from absl import logging
import numpy as np
import os
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader

from cabby.evals import utils as eu
from cabby.model.text import util
from cabby.model.landmark_recognition import run as run_pivot

criterion = nn.CosineEmbeddingLoss()


class Trainer:
  def __init__(
          self,
          model: torch.nn.Module,
          device: torch.device,
          optimizer: torch.optim.Adam,
          file_path: str,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          test_loader: DataLoader,
          unique_cells: Sequence[int],
          num_epochs: int,
          cells_tensor: torch.tensor,
          label_to_cellid: Dict[int, int],
          is_distance_distribution: bool

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
    self.best_valid_loss = float("Inf")
    if not os.path.exists(self.file_path):
      os.mkdir(self.file_path)
    self.model_path = os.path.join(self.file_path, 'model.pt')
    self.metrics_path = os.path.join(self.file_path, 'metrics.tsv')
    self.is_distance_distribution = is_distance_distribution


  def evaluate_landmark(
            self, 
            model_pivot: torch.nn.Module, 
            val_dataloader_geo: DataLoader, 
            val_dataloader_pivot: DataLoader):

    # Put the model into evaluation mode.
    self.model.eval() 
    model_pivot.eval()
    true_points_list, predictions_list, true_vals = [], [], []
    for batch_geo, batch_pivot in zip(val_dataloader_geo, val_dataloader_pivot):
      output = run_pivot.run_single(
        batch_pivot, model_pivot, self.device, is_train=False)
      
      hidden_states_pivot = output.hidden_states[-1][:,-1,:]
      text_geo = {key: val.to(self.device) for key, val in batch_geo['text'].items()}
      landmark_encode = self.model.landmark_main(hidden_states_pivot)
      cellid_embedding = self.model(self.cells_tensor)
      text_embedding = self.model.text_embed(text_geo)
      full_text_representation = torch.cat((text_embedding, landmark_encode), axis=-1)
      batch_dim = text_embedding.shape[0]
      cell_dim = cellid_embedding.shape[0]
      output_dim  = cellid_embedding.shape[1]
      cellid_embedding_exp = cellid_embedding.expand(
        batch_dim, cell_dim, output_dim)
      output = self.cos(
        cellid_embedding_exp, full_text_representation.unsqueeze(1))
      output = output.detach().cpu().numpy()
      predictions = np.argmax(output, axis=1)
      predictions_list.append(predictions)
      labels = batch_geo['label'].numpy()
      true_vals.append(labels)
      true_points_list.append(batch_geo['point'])

    true_points_list = np.concatenate(true_points_list, axis=0)
    predictions_list = np.concatenate(predictions_list, axis=0)
    pred_points_list = util.predictions_to_points(
      predictions_list, self.label_to_cellid)
    true_vals = np.concatenate(true_vals, axis=0)

    return (predictions_list, true_vals, 
    true_points_list, pred_points_list)


  def evaluate(self, validation_set: bool = True):
    '''Validate the model.'''

    if validation_set:
      data_loader = self.valid_loader
    else:
      data_loader = self.test_loader

    # Validation loop.
    logging.info("Starting evaluation.")
    true_points_list, pred_points_list = [], []
    predictions_list, true_vals  = [], []
    correct = 0
    total = 0
    loss_val_total = 0

    self.model.eval()
    with torch.no_grad():
      for batch in data_loader:

        text = {key: val.to(self.device) for key, val in batch['text'].items()}
        cellids = batch['cellid'].float().to(self.device)
        neighbor_cells = batch['neighbor_cells'].float().to(self.device) 
        far_cells = batch['far_cells'].float().to(self.device)

        loss = self.compute_loss(text, cellids, neighbor_cells, far_cells)
        loss_val_total+=loss
        cellid_embedding = self.model(self.cells_tensor)
        text_embedding = self.model.text_main(text)
        batch_dim = text_embedding.shape[0]
        cell_dim = cellid_embedding.shape[0]
        output_dim  = cellid_embedding.shape[1]
        cellid_embedding_exp = cellid_embedding.expand(
          batch_dim, cell_dim, output_dim)
        text_embedding_exp = text_embedding.unsqueeze(1)
        output = self.cos(cellid_embedding_exp, text_embedding_exp)
        if self.is_distance_distribution:
          prob = batch['prob'].float().to(self.device) 
          output = output*prob
        output = output.detach().cpu().numpy()
        predictions = np.argmax(output, axis=1)
        predictions_list.append(predictions)
        labels = batch['label'].numpy()
        true_vals.append(labels)
        true_points_list.append(batch['point'])

    true_points_list = np.concatenate(true_points_list, axis=0)
    predictions_list = np.concatenate(predictions_list, axis=0)
    pred_points_list = util.predictions_to_points(
      predictions_list, self.label_to_cellid)
    true_vals = np.concatenate(true_vals, axis=0)
    average_valid_loss = loss_val_total / len(data_loader)

    return (average_valid_loss, predictions_list, true_vals, 
    true_points_list, pred_points_list)
    
  def compute_embed(
          self,
          text: Dict, 
          cellids: torch.tensor, 
          neighbor_cells: torch.tensor, 
          far_cells: torch.tensor):

    text_embedding = self.model.text_embed(text)
    # Correct cellid.
    cellid_embedding = self.model(cellids)

    # Neighbor cellid.
    cellid_embedding_neighbor = self.model(neighbor_cells)

    # Far cellid.
    cellid_embedding_far = self.model(far_cells)

    return (
      text_embedding, 
      cellid_embedding, 
      cellid_embedding_neighbor, 
      cellid_embedding_far)

  def get_targets(self, cell_size):
    positive = torch.ones(cell_size).to(self.device)
    negative = -1*torch.ones(cell_size).to(self.device)
    return positive, negative

  def compute_loss_raw(
          self,
          text_embedding: torch.tensor, 
          cellid_embedding: torch.tensor, 
          cellid_embedding_neighbor: torch.tensor,
          cellid_embedding_far: torch.tensor,
          target_positive: torch.tensor, 
          target_negative: torch.tensor, 

):

    
    # Correct cellid.
    loss_cellid = criterion(text_embedding, cellid_embedding, target_positive)

    # Neighbor cellid.
    loss_neighbor = criterion(
      text_embedding, 
      cellid_embedding_neighbor, 
      target_negative)

    # Far cellid.
    loss_far = criterion(
      text_embedding, 
      cellid_embedding_far, 
      target_negative)

    return loss_cellid, loss_neighbor, loss_far
  

  def compute_loss(
          self,
          text: Dict, 
          cellids: torch.tensor, 
          neighbor_cells: torch.tensor, 
          far_cells: torch.tensor):
    
    (
      text_embedding, 
      cellid_embedding, 
      cellid_embedding_neighbor, 
      cellid_embedding_far) = self.compute_embed(
        text, cellids, neighbor_cells, far_cells)

    
    target_positive, target_negative = self.get_targets(cellids.shape[0])



    loss_cellid, loss_neighbor, loss_far = self.compute_loss_raw(
      text_embedding, 
      cellid_embedding, 
      cellid_embedding_neighbor,
      cellid_embedding_far,
      target_positive, 
      target_negative)

    loss = loss_cellid + loss_neighbor + loss_far

    return loss.mean()

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
        neighbor_cells = batch['neighbor_cells'].float().to(self.device) 
        far_cells = batch['far_cells'].float().to(self.device)

        loss = self.compute_loss(text, cellids, 
          neighbor_cells, far_cells)

        loss.backward()
        self.optimizer.step()

        # Update running values.
        running_loss += loss.item()
        global_step += 1


      # Evaluation step.
      valid_loss, predictions, true_vals, true_points, pred_points = self.evaluate()

      average_train_loss = running_loss / batch_idx
      accuracy = accuracy_score(true_vals, predictions)

      # Resetting running values.
      running_loss = 0.0

      logging.info('Epoch [{}/{}], Step [{}/{}], \
          Accuracy: {:.4f},Train Loss: {:.4f}, Valid Loss: {:.4f}'
            .format(epoch+1, self.num_epochs, global_step,
                self.num_epochs*len(self.train_loader), accuracy,
                average_train_loss, valid_loss))

      # Save model and results in checkpoint.
      if self.best_valid_loss > valid_loss:
        self.best_valid_loss = valid_loss
        util.save_checkpoint(self.model_path, self.model, self.best_valid_loss)

      self.model.train()


    logging.info('Finished Training.')

    model_state = util.load_checkpoint(self.model_path, self.model, self.device)
    valid_loss = model_state['valid_loss']

    logging.info(
      f'Loaded best model (with validation loss {valid_loss}) for testing.')
    
    logging.info('Start testing.')

    test_loss, predictions, true_vals, true_points, pred_points = self.evaluate(
      validation_set = False)

    util.save_metrics_last_only(
          self.metrics_path, 
          true_points, 
          pred_points)

    accuracy = accuracy_score(true_vals, predictions)

    evaluator = eu.Evaluator()
    error_distances = evaluator.get_error_distances(self.metrics_path)
    _, mean_distance, median_distance, max_error, norm_auc = evaluator.compute_metrics(error_distances)

    logging.info(f"Test Accuracy: {accuracy},\
          Mean distance: {mean_distance}, \
          Median distance: {median_distance}, \
          Max error: {max_error}, \
          Norm AUC: {norm_auc}")

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