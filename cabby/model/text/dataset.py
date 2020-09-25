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
import torch
import os
from absl import logging
from transformers import BertTokenizer


MAX_SEQ_LEN = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def tokenize(instruction):
    tokenize = tokenizer.encode(
        instruction,
        max_length=MAX_SEQ_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        add_special_tokens=True,)
    return tokenize


def create_dataset(data_dir, batch_sizes, device):
    # Load datasets.

    # print (PAD_INDEX,UNK_INDEX)

    TEXT = data.Field(
        use_vocab=False,
        tokenize=tokenize,
        batch_first=True,
        
    )
    LABELS = data.Field(
        sequential=False,
        preprocessing=lambda xs: 1 if xs == "manhattan" else 0,
        use_vocab=False, 
        batch_first=True, 
        dtype=torch.float
    )

    train_ds, valid_ds, test_ds = data.TabularDataset.splits(
        path=data_dir,
        format='tsv',
        skip_header=True,
        train='train.tsv',
        validation='dev.tsv',
        test='test.tsv',
        fields=[
            ('label', LABELS),
            ('instructions', TEXT)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_ds, valid_ds, test_ds), batch_sizes=batch_sizes, shuffle=True, device=device
    )

    logging.info('Data sample: %s', vars(train_ds[0]))

    return train_iter, val_iter, test_iter

    # for i in train_ds.examples:
    #   if len(i.instructions)>512:
    #       print (len(i.instructions))

    # # Print an example.
    # logging.info('Data sample: %s', vars(train_ds[0]))


#   #  print (vars(train_ds[0]))
#   #   print (train_ds.examples[0].text)

# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create_dataset("~/data/morp/morp-small", (16, 256, 256), device)
