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
from transformers import T5Tokenizer, T5ForConditionalGeneration


T5_TYPE = "t5-small"

class S2GenerationModel(nn.Module):
  def __init__(
    self, text_dim=768, hidden_dim=200, s2cell_dim=64, output_dim= 100):
    super(S2GenerationModel, self).__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(T5_TYPE)


  def loss(text, cellid):
      loss = self.model(input_ids=text, labels=labels).loss
      return loss



