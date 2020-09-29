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

"""Simple text classification example.
This script trains a text classifier on a dataset containing paragraphs that
are linked to s2cells. 

A sentence is encoded with an BERT and a binary prediction is made from CLS 
token embedding using an MLP.

Example command line call:
$ bazel-bin/cabby/model/text/s2cellid_prediction/model_trainer \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --region Pittsburgh \ 
  --s2_level 12 \
  --output_dir ~/tmp/output\

"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader

from cabby.model.text.s2cellid_prediction import train
from cabby.model.text.s2cellid_prediction import dataset

FLAGS = flags.FLAGS


flags.DEFINE_string("data_dir", None,
          "The directory from which to load the dataset.")
flags.DEFINE_enum(
  "region", None, ['Pittsburgh', 'Manhattan'],
  "Map areas: Manhattan or Pittsburgh.")
flags.DEFINE_integer("s2_level", None, "S2 level of the S2Cells.")
flags.DEFINE_string("output_dir", None,
          "The directory where the model and results will be save to.")

flags.DEFINE_float(
  'learning_rate', default=5e-5,
  help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_integer(
  'train_batch_size', default=4,
  help=('Batch size for training.'))

flags.DEFINE_integer(
  'test_batch_size', default=4,
  help=('Batch size for testing and validating.'))

flags.DEFINE_integer(
  'num_epochs', default=20,
  help=('Number of training epochs.'))

flags.DEFINE_integer(
  'eval_every', None,
  help=('Evaluation after a number of trained samples.'))

# Required flags.
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("s2_level")
flags.mark_flag_as_required("output_dir")


def main(argv):


  train_dataset, val_dataset, test_dataset, n_lables = dataset.create_dataset(
    FLAGS.data_dir, FLAGS.region, FLAGS.s2_level)

  logging.info("Number of lables: {}".format(n_lables))


  train_loader = DataLoader(
    train_dataset, batch_size=FLAGS.train_batch_size, shuffle=True)
  valid_loader = DataLoader(
    val_dataset, batch_size=FLAGS.test_batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=n_lables, return_dict=True)

  if torch.cuda.device_count() > 1:
    logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

  model.to(device)

  optimizer = AdamW(model.parameters(), lr=FLAGS.learning_rate)

  train.train_model(
    model=model,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    file_path=FLAGS.output_dir, 
    eval_every=len(train_dataset) // 10
    )


if __name__ == '__main__':
  app.run(main)





# device = torch.device(
#   'cuda') if torch.cuda.is_available() else torch.device('cpu')


# train_dataset, val_dataset, test_dataset, n_lables = dataset.create_dataset(
#   "~/data/wikigeo/pittsburgh", "Pittsburgh", 12)

# logging.info("Number of lables: {}".format(n_lables))

# train_loader = DataLoader(
#   train_dataset, batch_size=2, shuffle=True)
# valid_loader = DataLoader(
#   val_dataset, batch_size=4, shuffle=False)


# model = DistilBertForSequenceClassification.from_pretrained(
#   'distilbert-base-uncased', return_dict=True, num_labels=n_lables)

# if torch.cuda.device_count() > 1:
#   logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
#   model = nn.DataParallel(model)

# model.to(device)

# optimizer = AdamW(model.parameters(), lr=5e-5)

# train.train_model(
#   model=model,
#   device=device,
#   num_epochs=2,
#   optimizer=optimizer,
#   train_loader=train_loader,
#   valid_loader=valid_loader,
#   file_path="~/tmp/output", 
#   eval_every=len(train_dataset) // 10
#   )