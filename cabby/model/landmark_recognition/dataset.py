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

from cabby.model import datasets


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
MAX_LEN = 256



def create_dataset(
  data_dir: str,
  region: str,
  s2level: int,
):
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
    region: The region of the data.
    s2level: The s2level of the cells.
  Returns:
    The train, validate and test sets and the dictionary of labels to cellids.
  '''
  rvs_dataset = datasets.RVSDataset(data_dir, s2level, region)

  train_dataset = TextGeoSplit(
    rvs_dataset.train)
  logging.info(
    f"Finished to create the train-set with {len(train_dataset)} samples")
  val_dataset = TextGeoSplit(
    rvs_dataset.valid)
  logging.info(
    f"Finished to create the valid-set with {len(val_dataset)} samples")
  test_dataset = TextGeoSplit(
    rvs_dataset.test)
  logging.info(
    f"Finished to create the test-set with {len(test_dataset)} samples")

  return train_dataset, val_dataset, test_dataset

tag_values = ['O', "L"]
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}

class TextGeoSplit(torch.utils.data.Dataset):
  """A split of of the RUN dataset.

  `points`: The ground true end-points of the samples.
  `labels`: The ground true label of the cellid.
  `cellids`: The ground truth S2Cell id.
  `neighbor_cells`: One neighbor cell id of the ground truth S2Cell id.
  `far_cells`: One far away cell id (in the region defined) of the ground truth
  'dprob': Gamma distribution probability.
  S2Cell id.
  """

  def __init__(self, data: pd.DataFrame):
    # Tokenize instructions.

    self.inputs = [
      tokenize_and_preserve_labels(sent, labs)
      for sent, labs in zip(data.instructions.tolist(), data.entity_span)
    ]
    self.sent = data.instructions.tolist()


  def __getitem__(self, idx: int):
    '''Supports indexing such that TextGeoDataset[i] can be used to get
    i-th sample.
    Arguments:
      idx: The index for which a sample from the dataset will be returned.
    Returns:
      A single sample including text, the correct cellid, a neighbor cellid,
      a far cellid, a point of the cellid and the label of the cellid.
    '''

    input = {k: torch.tensor(v) for k, v in self.inputs[idx].items()}
    input['instructions'] = self.sent[idx]
    return input

  def __len__(self):
    return len(self.inputs)


def tokenize_and_preserve_labels(sentence, text_labels):
  labels = []
  spans = sorted(list(text_labels.values()))

  sentence_words = word_tokenize(sentence)
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

  tokenized_input = tokenize_and_align_labels(labels, sentence_words)
  return tokenized_input

def tokenize_and_align_labels(tags, sentence_word, label_all_tokens = True):
  tokenized_inputs = tokenizer(sentence_word, truncation=True, is_split_into_words=True)
  word_ids = tokenized_inputs.word_ids(batch_index=0)
  previous_word_idx = None
  label_ids = []
  for word_idx in word_ids:
    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
    # ignored in the loss function.
    if word_idx is None:
      label_ids.append(-100)
    # We set the label for the first token of each word.
    elif word_idx != previous_word_idx:
      label_ids.append(tags[word_idx])
    # For the other tokens in a word, we set the label to either the current label or -100, depending on
    # the label_all_tokens flag.
    else:
      label_ids.append(tags[word_idx] if label_all_tokens else -100)
    previous_word_idx = word_idx

  assert len(label_ids) == len(tokenized_inputs['input_ids'])
  tokenized_inputs["labels"] = label_ids
  return tokenized_inputs



class PadSequence:
    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x['input_ids'].shape[0], reverse=True)
        # Get each sequence and pad it
        input_ids = [x['input_ids'] for x in sorted_batch]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)

        instructions = [x['instructions'] for x in sorted_batch]


        lengths = [len(x) for x in input_ids_padded]

        lengths = torch.LongTensor(lengths)
        labels = [torch.tensor(x['labels']).clone().detach() for x in sorted_batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        attention_masks = torch.tensor([[float(i != 0.0) for i in ii] for ii in input_ids_padded])

        sample = {'labels': labels_padded,
                  'input_ids': input_ids_padded,
                  'attention_masks': attention_masks,
                  'length': lengths,
                  'instructions': instructions
                  }
        return sample