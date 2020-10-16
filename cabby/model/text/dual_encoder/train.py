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


from typing import Tuple, Sequence, Optional, Dict, Text, Any

from absl import logging
import numpy as np
import os
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader

from cabby.model.text import util

criterion = nn.CosineEmbeddingLoss()


def evaluate(model: torch.nn.Module,
       valid_loader: DataLoader,
       device: torch.device,
       tensor_cells: torch.tensor,
       label_to_cellid: Dict[int, int]
       ):
  '''Validate the model.'''

  model.eval()

  cos = nn.CosineSimilarity(dim=2)

  # Validation loop.
  logging.info("Starting evaluation.")
  true_points_list, pred_points_list, predictions_list, true_vals = [], [], [], []
  correct = 0
  total = 0
  loss_val_total = 0

  for batch in valid_loader:
    text = {key: val.to(device) for key, val in batch['text'].items()}
    cellids = batch['cellid'].float().to(device)
    neighbor_cells = batch['neighbor_cells'].float().to(device) 
    far_cells = batch['far_cells'].float().to(device)

    # text, cellids, neighbor_cells, far_cells, points, labels = batch
    cellids, neighbor_cells, far_cells = cellids.float().to(device), neighbor_cells.float().to(device), far_cells.float().to(device)
    text = {key: val.to(device) for key, val in text.items()}

    loss = compute_loss(device, model, text, cellids, 
      neighbor_cells, far_cells)

    loss_val_total += loss.item()

    tensor_cells = tensor_cells.float().to(device)
    text_embedding, cellid_embedding = model(text, tensor_cells)
    batch_dim = text_embedding.shape[0]
    cell_dim = cellid_embedding.shape[0]
    output_dim  = cellid_embedding.shape[1]
    cellid_embedding_exp = cellid_embedding.expand(batch_dim,cell_dim,output_dim)
    text_embedding_exp = text_embedding.unsqueeze(1)
    output = cos(cellid_embedding_exp,text_embedding_exp)
    output = output.detach().cpu().numpy()
    predictions = np.argmax(output, axis=1)
    predictions_list.append(predictions)
    labels = batch['label'].numpy()
    true_vals.append(labels)
    true_points_list.append(batch['point'])


  true_points_list = np.concatenate(true_points_list, axis=0)
  predictions_list = np.concatenate(predictions_list, axis=0)
  pred_points_list = util.predictions_to_points(predictions_list, label_to_cellid)
  true_vals = np.concatenate(true_vals, axis=0)
  average_valid_loss = loss_val_total / len(valid_loader)


  return (loss_val_total, predictions_list, true_vals, true_points_list, pred_points_list)

def compute_loss(device: torch.device, model: torch.nn.Module, 
  text: Dict, cellids: torch.tensor, neighbor_cells: torch.tensor, 
  far_cells: torch.tensor):
  
  # Correct cellid.
  target = torch.ones(cellids.shape[0]).to(device)
  text_embedding, cellid_embedding = model(text, cellids)
  loss_cellid = criterion(text_embedding, cellid_embedding, target)

  # Neighbor cellid.
  target_neighbor = -1*torch.ones(cellids.shape[0]).to(device)
  text_embedding_neighbor, cellid_embedding = model(text, neighbor_cells)
  loss_neighbor = criterion(text_embedding_neighbor, 
  cellid_embedding, target_neighbor)

  # Far cellid.
  target_far = -1*torch.ones(cellids.shape[0]).to(device)
  text_embedding_far, cellid_embedding = model(text, far_cells)
  loss_far = criterion(text_embedding_far, cellid_embedding, target_far)

  loss = loss_cellid + loss_neighbor + loss_far

  return loss.mean()



def train_model(model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Adam,
        file_path: Text,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        unique_cells: Sequence[int],
        num_epochs: int,
        cells_tensor: torch.tensor,
        label_to_cellid: Dict[int, int],
        best_valid_loss: float = float("Inf")):

  '''Main function for training model.'''
  # Initialize running values.
  running_loss = 0.0
  global_step = 0
  valid_accuracy_list, train_loss_list, valid_loss_list = [], [], []
  global_steps_list, true_points_list, pred_points_list = [], [], []

  # Training loop.
  model.train()

  for epoch in range(num_epochs):
    logging.info("Epoch number: {}".format(epoch))
    for batch_idx, batch in enumerate(train_loader):
      optimizer.zero_grad()
      text = {key: val.to(device) for key, val in batch['text'].items()}
      cellids = batch['cellid'].float().to(device)
      neighbor_cells = batch['neighbor_cells'].float().to(device) 
      far_cells = batch['far_cells'].float().to(device)

      loss = compute_loss(device, model, text, cellids, 
        neighbor_cells, far_cells)

      loss.backward()
      optimizer.step()

      # Update running values.
      running_loss += loss.mean().item()
      global_step += 1

    # Evaluation step.
    valid_loss, predictions, true_vals, true_points, pred_points = evaluate(
      model= model, valid_loader=valid_loader,device=device, 
      tensor_cells=cells_tensor, label_to_cellid=label_to_cellid)

    average_train_loss = running_loss / batch_idx
    accuracy = accuracy_score(true_vals, predictions)
    train_loss_list.append(average_train_loss)
    global_steps_list.append(global_step)
    valid_accuracy_list.append(accuracy)
    true_points_list.append(true_points)
    pred_points_list.append(pred_points)
    valid_loss_list.append(valid_loss)

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
      util.save_checkpoint(os.path.join(file_path, 'model.pt'), 
        model, best_valid_loss)
      util.save_metrics(os.path.join(file_path, 'metrics.pt'), train_loss_list,
        valid_loss_list, global_steps_list, valid_accuracy_list, 
        true_points_list, pred_points_list)

      model.train()

  logging.info('Finished Training.')
