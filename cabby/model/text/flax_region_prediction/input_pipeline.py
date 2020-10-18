# # Copyright 2020 The Flax Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """SST-2 input pipeline."""

# # pylint: disable=too-many-arguments,import-error,too-many-instance-attributes,too-many-locals
# import collections
# import re
# import os
# from typing import Dict, Sequence, Text

# from absl import logging

# import numpy as np
# import tensorflow as tf
# import tensorflow_datasets as tfds

# def build_vocab(datasets: Sequence[tf.data.Dataset],
#                 special_tokens: Sequence[Text] = (b'<pad>', b'<unk>', b'<s>', b'</s>'),
#                 min_freq: int = 0) -> Dict[Text, int]:
#   """Returns a vocabulary of tokens with optional minimum frequency."""
#   # Count the tokens in the datasets.
#   counter = collections.Counter()
#   for dataset in datasets:
#     for example in tfds.as_numpy(dataset):
#       counter.update(whitespace_tokenize(example['sentence']))

#   # Add special tokens to the start of vocab.
#   vocab = collections.OrderedDict()
#   for token in special_tokens:
#     vocab[token] = len(vocab)

#   # Add all other tokens to the vocab if their frequency is >= min_freq.
#   for token in sorted(list(counter.keys())):
#     if counter[token] >= min_freq:
#       vocab[token] = len(vocab)

#   logging.info('Number of unfiltered tokens: %d', len(counter))
#   logging.info('Vocabulary size: %d', len(vocab))
#   return vocab


# def whitespace_tokenize(text: Text) -> Sequence[Text]:
#   """Splits an input into tokens by whitespace."""
#   return text.strip().split()


# def get_shuffled_batches(dataset: tf.data.Dataset,
#                          seed: int = 0,
#                          batch_size: int = 64) -> tf.data.Dataset:
#   """Returns a Dataset that consists of padded batches when iterated over.
#   This shuffles the examples randomly each epoch. The random order is
#   deterministic and controlled by the seed.
#   Batches are padded because sentences have different lengths.
#   Sentences that are shorter in a batch will get 0s added at the end, until
#   all sentences in the batch have the same length.
#   Args:
#     dataset: A TF Dataset with examples to be shuffled and batched.
#     seed: The seed that determines the shuffling order, with a different order
#       each epoch.
#     batch_size: The size of each batch. The remainder is dropped.
#   Returns:
#     A TF Dataset containing padded batches.
#   """
#   # For shuffling we need to know how many training examples we have.
#   num_examples = dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

#   # `padded_shapes` says what kind of shapes to expect: [] means a scalar, [-1]
#   # means a vector of variable length, and [1] means a vector of size 1.
#   return dataset.shuffle(
#       num_examples, seed=seed, reshuffle_each_iteration=True).padded_batch(
#           batch_size,
#           padded_shapes={
#               'idx': [],
#               'sentence': [-1],
#               'label': [1],
#               'length': []
#           },
#           drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)


# def get_batches(dataset: tf.data.Dataset,
#                 batch_size: int = 64) -> tf.data.Dataset:
#   """Returns a Dataset that consists of padded batches when iterated over."""
#   return dataset.padded_batch(
#       batch_size,
#       padded_shapes={
#           'idx': [],
#           'sentence': [-1],
#           'label': [1],
#           'length': []
#       },
#       drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)

# def _get_label_index(label):
#   """Assign index 0 to pittsburgh and 1 to manhattan."""
#   return tf.cast(label == 'manhattan', tf.int32)

# def _parse_line_as_example(line):
#   """Parses line with 'label<TAB>paragraph' into separate label and text."""
#   tabsep_strings = tf.strings.split(line, maxsplit=1)
#   label = tabsep_strings[0]
#   text = tabsep_strings[1]
#   return {
#     'idx': tf.strings.to_hash_bucket(line, 100000),
#     'label': _get_label_index(label),
#     'sentence': text
#   }

# def create_dataset_from_tsv(datafile):
#   """Produce dataset given one-example-per-line text file."""
#   dataset = tf.data.TextLineDataset(datafile)
#   dataset = dataset.map(lambda x: _parse_line_as_example(x))
#   return dataset

# class MorpDataSource:
#   """Provides MORP data as pre-processed batches, a vocab, and embeddings."""
#   # pylint: disable=too-few-public-methods

#   def __init__(self, data_dir, min_freq: int = 0):
#     # Load datasets.
#     train_raw = create_dataset_from_tsv(os.path.join(data_dir, 'train.tsv'))
#     valid_raw = create_dataset_from_tsv(os.path.join(data_dir, 'dev.tsv'))
#     test_raw = create_dataset_from_tsv(os.path.join(data_dir, 'test.tsv'))

#     # Print an example.
#     logging.info(
#         'Data sample: %s', next(iter(tfds.as_numpy(train_raw.skip(4)))))

#     # Get a vocabulary and a corresponding GloVe word embedding matrix.
#     vocab = build_vocab((train_raw,), min_freq=min_freq)

#     unk_idx = vocab[b'<unk>']
#     bos_idx = vocab[b'<s>']
#     eos_idx = vocab[b'</s>']

#     # Turn data examples into pre-processed examples by turning each sentence
#     # into a sequence of token IDs. Also pre-prepend a beginning-of-sequence
#     # token <s> and append an end-of-sequence token </s>.

#     def tokenize(text: tf.Tensor):
#       """Whitespace tokenize text."""
#       return [whitespace_tokenize(text.numpy())]

#     def tf_tokenize(text: tf.Tensor):
#       return tf.py_function(tokenize, [text], Tout=tf.string)

#     def encode(tokens: tf.Tensor):
#       """Encodes a sequence of tokens (strings) into a sequence of token IDs."""
#       return [[vocab[t] if t in vocab else unk_idx for t in tokens.numpy()]]

#     def tf_encode(tokens: tf.Tensor):
#       """Maps tokens to token IDs."""
#       return tf.py_function(encode, [tokens], Tout=tf.int64)

#     def tf_wrap_sequence(sequence: tf.Tensor):
#       """Prepends BOS ID and appends EOS ID to a sequence of token IDs."""
#       return tf.concat(([bos_idx], tf.concat((sequence, [eos_idx]), 0)), 0)

#     def preprocess_example(example: Dict[Text, tf.Tensor]):
#       example['sentence'] = tf_wrap_sequence(
#           tf_encode(tf_tokenize(example['sentence'])))
#       example['label'] = [example['label']]
#       example['length'] = tf.shape(example['sentence'])[0]
#       return example

#     self.preprocess_fn = preprocess_example

#     # Pre-process all datasets.
#     self.train_dataset = train_raw.map(preprocess_example).cache()
#     self.valid_dataset = valid_raw.map(preprocess_example).cache()
#     self.test_dataset = test_raw.map(preprocess_example).cache()

#     self.valid_raw = valid_raw
#     self.test_raw = test_raw

#     self.vocab = vocab
#     self.vocab_size = len(vocab)

#     self.unk_idx = unk_idx
#     self.bos_idx = bos_idx
#     self.eos_idx = eos_idx