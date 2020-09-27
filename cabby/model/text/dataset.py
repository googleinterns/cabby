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

from torchtext import data
import torch
import os
from absl import logging
import transformers
import pandas as pd

from transformers import DistilBertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


class CabbyDataset(torch.utils.data.Dataset):
  def __init__(self, encodings: BatchEncoding , labels: Sequence[int]):

    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx: int):
    item = {key: torch.tensor(val[idx])
                              for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def create_dataset(data_dir: Text) -> Tuple[CabbyDataset, CabbyDataset, CabbyDataset]:
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
  Returns:
    The train, validate and test sets.
  '''

  LABELS = data.Field(
      sequential=False,
      preprocessing=lambda xs: 1 if xs == "manhattan" else 0,
      use_vocab=False, 
      batch_first=True, 
  )
  TEXT = data.Field(
      use_vocab=False,
      batch_first=True,
      sequential=False,  
  )

  train_ds, valid_ds, test_ds = data.TabularDataset.splits(
      path=data_dir,
      format='tsv',
      skip_header=False,
      train='train.tsv',
      validation='dev.tsv',
      test='test.tsv',
      fields=[
          ('label', LABELS),
          ('instructions', TEXT)])

  logging.info('Data sample: %s', vars(train_ds[0]))

  # Get list of instructions.
  train_texts = [train_ds.examples[idx].instructions for idx in range(len(train_ds))]
  val_texts = [valid_ds.examples[idx].instructions for idx in range(len(valid_ds))]
  test_texts = [test_ds.examples[idx].instructions for idx in range(len(test_ds))]

  # Get list of lables.
  train_labels = [train_ds.examples[idx].label for idx in range(len(train_ds))]
  val_labels = [valid_ds.examples[idx].label for idx in range(len(valid_ds))]
  test_labels = [test_ds.examples[idx].label for idx in range(len(test_ds))]


  # Tokenize instructions.
  tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
  train_encodings = tokenizer(train_texts, truncation=True, padding=True, add_special_tokens=True)
  val_encodings = tokenizer(val_texts, truncation=True, padding=True, add_special_tokens=True)
  test_encodings = tokenizer(test_texts, truncation=True, padding=True, add_special_tokens=True)

  # Create Cabby dataset.
  train_dataset = CabbyDataset(train_encodings, train_labels)
  val_dataset = CabbyDataset(val_encodings, val_labels)
  test_dataset = CabbyDataset(test_encodings, test_labels)

  return train_dataset, val_dataset, test_dataset


