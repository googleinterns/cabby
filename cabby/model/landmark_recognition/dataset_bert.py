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

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


class EntityRecognitionSplit(torch.utils.data.Dataset):
  """A split of the Entity Recognition dataset ."""

  def __init__(self, data: pd.DataFrame):
    # Tokenize instructions and corresponding labels.
    basic_tokenization = [
      basic_tokenize_and_align_labels(sent, labs)
      for sent, labs in zip(data.instructions.tolist(), data.entity_span)
    ]
    self.inputs = [bert_tokenize_and_align_labels(sent, labs) for sent, labs in basic_tokenization]
    self.sent = data.instructions.tolist()


  def __getitem__(self, idx: int):
    '''Supports indexing such that TextGeoDataset[i] can be used to get
    i-th sample.
    Arguments:
      idx: The index for which a sample from the dataset will be returned.
    Returns:
      A single sample including text, the tokenization of the text and
      the corresponding labels.
    '''

    input = {k: torch.tensor(v) for k, v in self.inputs[idx].items()}
    input['instructions'] = self.sent[idx]
    return input

  def __len__(self):
    return len(self.inputs)

def create_dataset(
  data_dir: str,
  region: str,
  s2level: int,
) -> Tuple[EntityRecognitionSplit, EntityRecognitionSplit, EntityRecognitionSplit]:
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
    region: The region of the data.
    s2level: The s2level of the cells.
  Returns:
    The train, validate and test sets.
  '''
  rvs_dataset = datasets.RVSDataset(data_dir, s2level, region)

  train_dataset = EntityRecognitionSplit(
    rvs_dataset.train)
  logging.info(
    f"Finished to create the train-set with {len(train_dataset)} samples")
  val_dataset = EntityRecognitionSplit(
    rvs_dataset.valid)
  logging.info(
    f"Finished to create the valid-set with {len(val_dataset)} samples")
  test_dataset = EntityRecognitionSplit(
    rvs_dataset.test)
  logging.info(
    f"Finished to create the test-set with {len(test_dataset)} samples")

  return train_dataset, val_dataset, test_dataset

def basic_tokenize_and_align_labels(
  sentence: str, text_labels: Dict[str,Tuple[int, int]]
) :
  '''Tokenize sentence and preserve the labels,
  such that for each token there is a label.
  Arguments:
    sentence: the sentence to tokenize.
    text_labels: the dictionary containing the landmarks (keys) and span (values).
  Returns:
    A List of the tokens and their corresponding labels.
  '''

  # Basic tokenization of the sentence.
  sentence_words = word_tokenize(sentence)

  # Create labels for each token according to the dictionary of entities and spans.
  labels = []
  spans = sorted(list(text_labels.values()))

  cur_index = 0
  span = spans.pop(0)
  start, end = span[0], span[1]
  for word in sentence_words:
    if cur_index >= start:
      labels.append(1)
    else:
      labels.append(0)

    cur_index += len(word)
    if cur_index<len(sentence) and sentence[cur_index] == ' ':
      cur_index+=1
    if cur_index >= end:
      if len(spans)>0:
        span = spans.pop(0)
        start, end = span[0], span[1]
      else:
        start = len(sentence)+1

  return sentence_words, labels

def bert_tokenize_and_align_labels(
  tokenized_sentence: List[int], tags: List[int]
) -> Dict[str, torch.Tensor]:
  '''Bert-tokenization of sentence and preserve the labels,
  such that for each token there is a label.
  Arguments:
    tokenized_sentence: the tokenized sentence.
    tags: the labels corresponding to the tokens.
  Returns:
    A dictionary of Bert-tokenized sentences and corresponding labels.
  '''
  tokenized_inputs = tokenizer(tokenized_sentence, truncation=True, is_split_into_words=True)
  word_ids = tokenized_inputs.word_ids(batch_index=0)
  previous_word_idx = None
  label_ids = []
  for word_idx in word_ids:
    # Special tokens have a word id that is None.
    # We set the label to -100 so they are automatically
    # ignored in the loss function.
    if word_idx is None:
      label_ids.append(-100)
    # We set the label for the first token of each word.
    elif word_idx != previous_word_idx:
      label_ids.append(tags[word_idx])
    # For the other tokens in a word, we set the label to the current label,
    else:
      label_ids.append(tags[word_idx])
    previous_word_idx = word_idx

  assert len(label_ids) == len(tokenized_inputs['input_ids'])

  tokenized_inputs["labels"] = label_ids
  return tokenized_inputs


class PadSequence:
  def __call__(self, batch):
    sorted_batch = sorted(batch, key=lambda x: x['input_ids'].shape[0], reverse=True)

    # Get each sequence and pad it.
    input_ids = [x['input_ids'] for x in sorted_batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)

    labels = [torch.tensor(x['labels']).clone().detach() for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    attention_masks = torch.tensor(
      [[float(i != 0.0) for i in ii] for ii in input_ids_padded])

    sample = {'labels': labels_padded,
              'input_ids': input_ids_padded,
              'attention_mask': attention_masks,
              }
    return sample