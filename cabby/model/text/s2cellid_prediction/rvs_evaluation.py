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
This script Loads a pre-trained geomodel and evaluates the RVS samples against it. 

Example command line call:
$ bazel-bin/cabby/model/text/s2cellid_prediction/rvs_evaluation \
  --rvs_path ~/data/wikigeo/pittsburgh/rvs.json  \
  --model_path ~/tmp/output/pittsburgh/12/model.pt \
  --train_dataset_dir ~/model/wikigeo/dataset/pittsburgh \
  --s2_level 12

"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader

from cabby.model.text.s2cellid_prediction import train
from cabby.model.text.s2cellid_prediction import rvs_dataset

FLAGS = flags.FLAGS


flags.DEFINE_string("rvs_path", None,
          "The directory from which to load the RVS data.")
flags.DEFINE_string("model_path", None,
          "The path of the model.")

flags.DEFINE_integer(
  'batch_size', default=16,
  help=('Batch size for evaluation.'))

flags.DEFINE_string(
  'train_dataset_dir', None,
  help=('The directory of the trained dataset.'))

flags.DEFINE_integer("s2_level", None, "S2 level of the S2Cells.")


# Required flags.
flags.mark_flag_as_required("rvs_path")
flags.mark_flag_as_required("model_path")
flags.mark_flag_as_required("train_dataset_dir")
flags.mark_flag_as_required("s2_level")


def main(argv):

  if not os.path.exists(FLAGS.rvs_path):
    sys.exit("RVS path doesn't exsist: {}.".format(FLAGS.rvs_path))

  if not os.path.exists(FLAGS.model_path):
    sys.exit("Model path doesn't exsist: {}.".format(FLAGS.model_path))

  dataset_path = os.path.join(FLAGS.train_dataset_dir, str(FLAGS.s2_level))

  if not os.path.exists(dataset_path):
    sys.exit("Trained datset path doesn't exist: {}.".format(dataset_path))

  lables_dictionary_path = os.path.join(dataset_path, "label_to_cellid.npy")
  cellids_dictionary_path = os.path.join(dataset_path, "cellid_to_label.npy")

  logging.info("Preparing data.")

  valid_dataset, label_to_cellid = rvs_dataset.create_dataset(
    path=FLAGS.rvs_path,
    cellid_to_label_path=cellids_dictionary_path, lables_dictionary_path=lables_dictionary_path,
    s2level=FLAGS.s2_level
  )
  logging.info("Size of data: {}".format(len(valid_dataset)))


  n_lables = len(label_to_cellid)
  logging.info("Number of lables: {}".format(n_lables))

  valid_loader = DataLoader(
    valid_dataset, batch_size=FLAGS.batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

  model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=n_lables, return_dict=True)

  if torch.cuda.device_count() > 1:
    logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)

  logging.info("Loading model weights.")
  train.load_checkpoint(FLAGS.model_path, model, device)

  model.to(device)

  valid_loss, predictions, true_vals, true_points, pred_points = train.evaluate(
    model=model,
    device=device,
    valid_loader=valid_loader,
    label_to_cellid=label_to_cellid
  )
  # print (true_vals)
  accuracy = train.accuracy_cells(true_vals, predictions)
  logging.info('Accuracy: {:.4f}, Valid Loss: {:.4f}'
         .format(accuracy, valid_loss))


if __name__ == '__main__':
  app.run(main)
