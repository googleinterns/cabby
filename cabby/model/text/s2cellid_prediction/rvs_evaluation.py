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
  --model_path ~/tmp/output/model.pt \

"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import os 
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader

from cabby.model.text.s2cellid_prediction import train
from cabby.model.text.s2cellid_prediction import dataset

FLAGS = flags.FLAGS


flags.DEFINE_string("rvs_path", None,
          "The directory from which to load the RVS data.")
flags.DEFINE_string("model_path", None,
          "The path of the model.")

flags.DEFINE_integer(
  'batch_size', default=16,
  help=('Batch size for evaluation.'))

flags.DEFINE_integer(
  'label_dictionary_path', default=16,
  help=('The label_to_.'))

# Required flags.
flags.mark_flag_as_required("rvs_path")
flags.mark_flag_as_required("model_path")
flags.mark_flag_as_required("batch_size")

def main(argv):


  if not os.path.exists(FLAGS.rvs_path):
    sys.exit("RVS path doesn't exsist: {}.".format(FLAGS.rvs_path))

  if not os.path.exists(FLAGS.model_path):
    sys.exit("Model path doesn't exsist: {}.".format(FLAGS.model_path))

  logging.info("Preparing data.")
   = dataset.create_dataset(
    FLAGS.data_dir, FLAGS.region, FLAGS.s2_level)
  n_lables = len(label_to_cellid)
  logging.info("Number of lables: {}".format(n_lables))

    
    # Save to dataloaders and lables to cells dictionary.
    os.mkdir(dataset_path)
    torch.save(train_dataset, train_path_dataset)
    torch.save(valid_dataset, valid_path_dataset)
    np.save(lables_dictionary, label_to_cellid) 
    logging.info("Saved data.")

  train_loader = DataLoader(
  train_dataset, batch_size=FLAGS.train_batch_size, shuffle=True)
  valid_loader = DataLoader(
  valid_dataset, batch_size=FLAGS.test_batch_size, shuffle=False)


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
    label_to_cellid = label_to_cellid,
    file_path=FLAGS.output_dir, 
    eval_every=1000,
    )


if __name__ == '__main__':
  app.run(main)



