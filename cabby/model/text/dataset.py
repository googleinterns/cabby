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

"""Loads the dataset."""

from torchtext import data
import os
from absl import logging
from transformers import BertTokenizer




def create_dataset(data_dir, batch_sizes, device):
  # Load datasets.

  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  MAX_SEQ_LEN = 124
  PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
  UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


  TEXT =data.Field(use_vocab=False, tokenize=tokenizer.encode, fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
  LABELS = data.LabelField()


  train_ds, valid_ds, test_ds = data.TabularDataset.splits(
  path=data_dir, 
  format='tsv', 
  skip_header=True, 
  train='train.tsv', 
  validation='dev.tsv', 
  test='test.tsv',  
  fields=[
  ('labels', LABELS), 
  ('instructions', TEXT)])


  # # Get a vocabulary.
  LABELS.build_vocab(train_ds)

  train_iter, val_iter, test_iter = data.BucketIterator.splits(
  (train_ds, valid_ds, test_ds), batch_sizes=batch_sizes, shuffle=True, device=device
  )

  return train_iter, val_iter, test_iter
  


  # # Print an example.
  logging.info('Data sample: %s', vars(train_ds[0]))


#   #  print (vars(train_ds[0]))
#   #   print (train_ds.examples[0].text)

# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create_dataset("~/data/morp/morp-small", (16, 256, 256), device)
