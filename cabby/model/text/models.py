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

from absl import logging

import numpy as np
import sys
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model

from typing import Dict, Sequence 

from cabby.model import util as mutil
from cabby.geo import util as gutil


T5_TYPE = "t5-small"
T5_DIM = 512 if T5_TYPE=="t5-small" else 768
BERT_TYPE = "distilbert-base-uncased"

criterion = nn.CosineEmbeddingLoss()


class GeneralModel(nn.Module):
  def __init__(self, device):
    super(GeneralModel, self).__init__()
    self.is_generation = False
    self.device = device

  def get_embed(self, text_feat, cellid):
    return text_feat, cellid
  def forward(self, text: Dict, *args
  ):
    sys.exit("Implement compute_loss function in model")
 
  def predict(self, text, *args):
    sys.exit("Implement prediction function in model")


class DualEncoder(GeneralModel):
  def __init__(
    self, 
    device, 
    text_dim=768, 
    hidden_dim=200, 
    s2cell_dim=64, 
    output_dim=100, 
    ):
    GeneralModel.__init__(self, device)
    
    self.hidden_layer = nn.Linear(text_dim, hidden_dim)
    self.softmax = nn.Softmax(dim=-1)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
  
    self.model = DistilBertModel.from_pretrained(
      BERT_TYPE, return_dict=True)

    self.text_main = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            )
    self.cellid_main = nn.Sequential(
            nn.Linear(s2cell_dim, output_dim),
            )
    self.cos = nn.CosineSimilarity(dim=2)


  def get_embed(self, text_feat, cellid):
    
    text_embedding = self.text_embed(text_feat)
    cellid_embedding = self.cellid_main(cellid.float())
    
    return text_embedding.shape[0], text_embedding, cellid_embedding



  def predict(self, text, all_cells, *args):

    batch_dim, text_embedding_exp, cellid_embedding = self.get_embed(text, all_cells)
    cell_dim = cellid_embedding.shape[0]
    output_dim  = cellid_embedding.shape[1]

    text_embedding_exp = text_embedding_exp.unsqueeze(1).expand(batch_dim, cell_dim, output_dim)
    cellid_embedding_exp = cellid_embedding.expand(batch_dim, cell_dim, output_dim)
    
    label_to_cellid = args[0]
    assert cellid_embedding_exp.shape == text_embedding_exp.shape
    output = self.cos(cellid_embedding_exp, text_embedding_exp)
    output = output.detach().cpu().numpy()
    predictions = np.argmax(output, axis=1)

    points = mutil.predictions_to_points(predictions, label_to_cellid)
    return points


  def forward(self, text, cellid, *args
  ):

    batch = args[0]
    neighbor_cells = batch['neighbor_cells']
    far_cells = batch['far_cells']

    # Correct cellid.
    target = torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding, cellid_embedding = self.get_embed(text, cellid)
    loss_cellid = criterion(text_embedding, cellid_embedding, target)

    # Neighbor cellid.
    target_neighbor = -1*torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding_neighbor, cellid_embedding = self.get_embed(text, neighbor_cells)
    loss_neighbor = criterion(text_embedding_neighbor, 
    cellid_embedding, target_neighbor)

    # Far cellid.
    target_far = -1*torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding_far, cellid_embedding = self.get_embed(text, far_cells)
    loss_far = criterion(text_embedding_far, cellid_embedding, target_far)

    loss = loss_cellid + loss_neighbor + loss_far

    return loss.mean()
  

  def text_embed(self, text):
    outputs = self.model(**text)
    cls_token = outputs.last_hidden_state[:,-1,:]
    return self.text_main(cls_token)


class S2GenerationModel(GeneralModel):
  def __init__(
      self, label_to_cellid, device, is_landmarks=False, is_path=False, 
      is_warmup_start_end=False,
      is_warmup_ner_landmarks=False):
    GeneralModel.__init__(self, device)
    self.model = T5ForConditionalGeneration.from_pretrained(T5_TYPE)
    self.tokenizer = T5Tokenizer.from_pretrained(T5_TYPE)
    self.is_generation = True
    self.label_to_cellid = label_to_cellid
    self.is_landmarks = is_landmarks
    self.is_path = is_path
    self.is_warmup_start_end = is_warmup_start_end
    self.is_warmup_ner_landmarks = is_warmup_ner_landmarks

    self.max_size = len(str(len(label_to_cellid)))

    if self.is_landmarks:
      self.max_size = self.max_size*10

    if self.is_path:
      self.max_size = self.max_size*100

  def forward(self, text, cellid, *args):
    batch = args[0]

    input_ids = text['input_ids'] 
    attention_mask = text['attention_mask'] 

    if self.is_landmarks:
      labels = batch['landmarks'].long()
    elif self.is_path:
      labels = batch['route'].long()
    elif self.is_warmup_start_end:
      input_ids = batch['start_end_input_ids'] 
      attention_mask = batch['start_end_attention_mask']
      labels = batch['route_fixed'].long()
    elif self.is_warmup_ner_landmarks:
      labels = batch['landmarks_ner'].long()
    else:
      labels = cellid.long()

    output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
    return output.loss

  def get_embed(self, text, cellid):
    text_dim = text['input_ids'].shape[0]
    return text_dim, text, cellid


  def predict(self, text, *args):

    label_to_cellid = args[1]

    output_sequences = self.model.generate(
      **text, num_beams=2, 
      max_length=self.max_size+2, 
      min_length=1, 
      remove_invalid_values=True)

    prediction = self.tokenizer.batch_decode(
      output_sequences, skip_special_tokens=True)

    prediction_cellids = []

    for pred_raw in prediction:

      pred = pred_raw.split(";")[0].replace(" ", "")

      if not pred.isdigit():
        pred = 0
      label_int = int(pred)
      if label_int in label_to_cellid:
        cell_id = label_to_cellid[label_int]
      else:
        cell_id = label_to_cellid[0]
      prediction_cellids.append(cell_id)

    prediction_coords = gutil.get_center_from_s2cellids(prediction_cellids)
    
    return prediction_coords

  
class ClassificationModel(GeneralModel):
  def __init__(self, n_lables, device, hidden_dim = 200):
    GeneralModel.__init__(self, device)
    self.is_generation = True

    self.model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=n_lables, return_dict=True)

    self.criterion = nn.CrossEntropyLoss()

  def forward(self, text, cellid, *args):
    labels =  args[0]['label']

    outputs = self.model(
      input_ids = text['input_ids'], 
      attention_mask=text['attention_mask'], 
      labels=labels)

    return outputs.loss

  def predict(self, text, all_cells, *args):
    label_to_cellid = args[0]

    outputs = self.model(**text)
    logits = outputs.logits.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)

    points = mutil.predictions_to_points(predictions, label_to_cellid)
    return points