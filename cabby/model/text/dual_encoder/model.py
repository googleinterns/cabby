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

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DualEncoder(nn.Module):
  def __init__(self, text_dim=768, hidden_dim=200, s2cell_dim=64, 
    output_dim= 100):
    super(DualEncoder, self).__init__()

    self.hidden_layer = nn.Linear(text_dim, hidden_dim)
    self.softmax = nn.Softmax(dim=-1)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.bert = DistilBertModel.from_pretrained(
      "distilbert-base-uncased", return_dict=True)
    self.text_main = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            )
    self.cellid_main = nn.Sequential(
            nn.Linear(s2cell_dim, output_dim),
            )

  def forward(self, text_feat, cellid):
      
    outputs = self.bert(**text_feat)
    cls_token = outputs.last_hidden_state[:,-1,:]
    text_embedding = self.text_main(cls_token)
    cellid_embedding = self.cellid_main(cellid)
    
    return text_embedding, cellid_embedding