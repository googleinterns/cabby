
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

criterion  = torch.nn.CosineEmbeddingLoss




def evaluate(model: torch.nn.Module,
       valid_loader: DataLoader,
       device: torch.device,
       tensor_cells: torch.tensor
       ):
  '''Validate the modal.'''

  model.eval()

  loss_val_total = 0

  cos = nn.CosineSimilarity(dim=2)

  # Validation loop.
  logging.info("Starting evaluation.")
  true_points_list, pred_points_list, predictions, true_vals = [], [], [], []
  correct = 0
  total = 0
  for batch in valid_loader:
    text, _, _, _, _, labels = batch
    text = {key: val.to(device) for key, val in text.items()}
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
    lables = labels.numpy()
    correct += (lables==predictions).sum()
    total += lables.shape[0]
  logging.info("Accuracy: {}, number examples: {}, correct: {}".format(round(100*correct/total, 3), total, correct))

  return


def train_model(model: torch.nn.Module,
        device: torch.device,
        optimizer: Any,
        file_path: Text,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        unique_cells: Any,
        num_epochs: int,
        tens_cells: torch.tensor,
        best_valid_loss: float = float("Inf"),
        ):
  '''Main funcion for training model.'''
  # initialize running values
  running_loss = 0.0
  global_step = 0
  valid_accuracy_list, train_loss_list, valid_loss_list = [], [], []
  global_steps_list, true_points_list, pred_points_list = [], [], []

  # Training loop.
  model.train()
  criterion = nn.CosineEmbeddingLoss()


  for epoch in range(num_epochs):
    running_loss = 0.0
    logging.info("Epoch number: {}".format(epoch))
    for batch_idx, batch in enumerate(train_loader):
      optimizer.zero_grad()
      text, cellids, neighbor_cells, far_cells, points, labels = batch
      text = {key: val.to(device) for key, val in text.items()}
      cellids, neighbor_cells, far_cells = cellids.float().to(device), neighbor_cells.float().to(device), far_cells.float().to(device)

      # Correct cellid.
      target = torch.ones(cellids.shape[0]).to(device)
      text_embedding, cellid_embedding = model(text, cellids)
      loss_cellid = criterion(text_embedding, cellid_embedding, target)

      # Neighbor cellid.
      target = -1*torch.ones(cellids.shape[0]).to(device)
      text_embedding, cellid_embedding = model(text, neighbor_cells)
      loss_neighbor = criterion(text_embedding, cellid_embedding, target)

      # Far cellid.
      text_embedding, cellid_embedding = model(text, far_cells)
      loss_far = criterion(text_embedding, cellid_embedding, target)

      loss = loss_cellid + loss_neighbor + loss_far
      loss.mean().backward()
      optimizer.step()

      # Update running values.
      running_loss += loss.mean().item()
      global_step += 1


    # Evaluation step.

    average_train_loss = running_loss / len(train_loader)
    logging.info("Loss: {}".format(average_train_loss))
    evaluate(model= model, valid_loader=valid_loader,device=device, tensor_cells=tens_cells)

    model.train()

  logging.info('Finished Training.')
