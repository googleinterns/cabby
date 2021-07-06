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
from nltk.tokenize import word_tokenize
import pandas as pd
import torch
from transformers import AutoTokenizer
from typing import Dict, Tuple, Any, List

from cabby.model import datasets

tokenizer = AutoTokenizer.from_pretrained("t5-small", padding=True, truncation=True)

EXTRACT_ALL_PIVOTS = "all"


class EntityRecognitionSplit(torch.utils.data.Dataset):
  """A split of the Entity Recognition dataset ."""

  def __init__(self, data: pd.DataFrame, pivot_type: str):
    # Tokenize instructions and corresponding labels.
    self.ds = data

    if pivot_type != EXTRACT_ALL_PIVOTS:
      self.ds['pivot_span'] = self.ds.apply(get_pivot_span_by_name, args=(pivot_type,), axis=1)
      self.labels = self.ds.pivot_span
    else:
      self.labels = self.ds.entity_span

    self.target_text = [', '.join(list(x.keys())).replace("_", " ").lower() for x in self.labels.tolist()]

    self.input_text = self.ds.instructions.tolist()

    self.inputs = tokenizer(self.input_text, padding=True, truncation=True, return_tensors="pt")
    self.targets = tokenizer(self.target_text, padding=True, truncation=True, return_tensors="pt")

  def __getitem__(self, index: int):

      source_ids  = self.inputs["input_ids"][index].squeeze().cuda()
      target_ids  = self.targets["input_ids"][index].squeeze().cuda()
      src_mask    = self.inputs["attention_mask"][index].squeeze().cuda()
      target_mask = self.targets["attention_mask"][index].squeeze().cuda()
      target_text = self.target_text[index]
      input_text = self.input_text[index]

      sample = {'source_ids': source_ids.cuda(),
                'target_ids': target_ids.cuda(),
                'src_mask': src_mask,
                'target_mask': target_mask,
                'target_text': target_text,
                'input_text': input_text}

      return sample

  def __len__(self):
    return len(self.input_text)

def get_pivot_span_by_name(sample: pd.Series, pivot_type: str
                           ) -> Dict[str, List[int]]:
  '''Get the entity span for a specific sample and a specific type of entity.
  Arguments:
    sample: the sample from which the span should be extracted.
    pivot_type: the type of the pivot.
  Returns:
    A span of an entity includes a start and end of the span positions.
  '''
  pivot_name = sample[pivot_type][2]
  if pivot_name:
    return {pivot_type: sample.entity_span[pivot_name]}
  return {pivot_type: [0, 0]}  # The pivot doesn't appear in the instructions.


def create_dataset(
  data_dir: str,
  region: str,
  s2level: int,
  pivot_type: str = EXTRACT_ALL_PIVOTS
) -> Tuple[EntityRecognitionSplit, EntityRecognitionSplit, EntityRecognitionSplit]:
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
    region: The region of the data.
    s2level: The s2level of the cells.
    pivot_type: name of the pivot to be extracted.
  Returns:
    The train, validate and test sets.
  '''
  rvs_dataset = datasets.RVSDataset(data_dir, s2level, region)
  train_dataset = EntityRecognitionSplit(rvs_dataset.train, pivot_type)
  logging.info(
    f"Finished to create the train-set with {len(train_dataset)} samples")
  val_dataset = EntityRecognitionSplit(rvs_dataset.valid, pivot_type)
  logging.info(
    f"Finished to create the valid-set with {len(val_dataset)} samples")
  test_dataset = EntityRecognitionSplit(rvs_dataset.test, pivot_type)
  logging.info(
    f"Finished to create the test-set with {len(test_dataset)} samples")

  return train_dataset, val_dataset, test_dataset
