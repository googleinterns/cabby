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

"""Model framework for text and S2Cellids matching.

Example command line call:
$ bazel-bin/cabby/model/text/model_trainer \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dataset/pittsburgh \
  --region Pittsburgh \ 
  --s2_level 12 \
  --output_dir ~/tmp/output/\
  --train_batch_size 32 \
  --test_batch_size 32 \

For infer:
$ bazel-bin/cabby/model/text/model_trainer \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dataset/pittsburgh \
  --region Pittsburgh \
  --s2_level 12 \
  --test_batch_size 32 \
  --infer_only True \
  --model_path ~/tmp/model/ \
  --output_dir ~/tmp/output/\
  --task RVS



"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import os 
import sys
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW

from cabby.evals import utils as eu
from cabby.model.text import train
from cabby.model import dataset_item
from cabby.model.text import models
from cabby.model import datasets
from cabby.model import util
# from cabby.geo import regions

TASKS = ["WikiGeo", "RVS", "RUN", "human"]

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("dataset_dir_train", None,
          "The directory from which to load the dataset for train.")


flags.DEFINE_string("dataset_dir_test", None,
          "The directory from which to load the dataset for test.")


# flags.DEFINE_enum(
#   "region", None, regions.SUPPORTED_REGION_NAMES,
#   regions.REGION_SUPPORT_MESSAGE)
  


# flags.DEFINE_integer("s2_level", None, "S2 level of the S2Cells.")
flags.DEFINE_string("output_dir", None,
          "The directory where the model and results will be save to.")
flags.DEFINE_float(
  'learning_rate', default=5e-5,
  help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_string("model_path", None,
          "A path of a model the model to be fine tuned\ evaluated.")


flags.DEFINE_integer(
  'train_batch_size', default=4,
  help=('Batch size for training.'))

flags.DEFINE_integer(
  'test_batch_size', default=4,
  help=('Batch size for testing and validating.'))

flags.DEFINE_integer(
  'num_epochs', default=5,
  help=('Number of training epochs.'))

flags.DEFINE_bool(
  'infer_only', default=False,
  help=('Train and infer\ just infer.'))

flags.DEFINE_bool(
  'is_single_sample_train', default=False,
  help=('Train on a single sample and do not evaluate.'))


flags.DEFINE_bool(
  'is_val_loss_from_model', default=False,
  help=('In case the model is loaded - should the validation loss use the models current loss.'))



# Required flags.
flags.mark_flag_as_required("dataset_dir_train")
flags.mark_flag_as_required("dataset_dir_test")



def main(argv):

  trainer_loader_list = []
  model_types = []
  for dataset_model_path in FLAGS.dataset_dir_train:
    if not os.path.isdir(dataset_model_path):
      sys.exit(f"The directory {dataset_model_path} does not exits")

    dataset_train = dataset_item.TextGeoDataset.load(
      dataset_dir=dataset_model_path
    )
    model_types.append(dataset_train.train.model_type)

    train_loader = DataLoader(
      dataset_train.train, batch_size=FLAGS.train_batch_size, shuffle=True)

    trainer_loader_list.append(train_loader)

  if not os.path.isdir(FLAGS.dataset_dir_test):
    sys.exit(f"The directory {FLAGS.dataset_dir_test} does not exits")

  dataset_valid_test = dataset_item.TextGeoDataset.load(
    dataset_dir = FLAGS.dataset_dir_test
  )

  valid_loader = DataLoader(
    dataset_valid_test.valid, batch_size=FLAGS.train_batch_size, shuffle=True)

  test_loader = DataLoader(
    dataset_valid_test.test, batch_size=FLAGS.train_batch_size, shuffle=True)


  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  run_model = models.S2GenerationModel(
      dataset_valid_test.label_to_cellid ,device=device)

  run_model.to(device)

  optimizer = torch.optim.Adam(
    run_model.parameters(), lr=FLAGS.learning_rate)
  
  run_model.best_valid_loss = float("Inf")


  trainer = train.Trainer(
    model=run_model,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer,
    train_loader=trainer_loader_list,
    valid_loader=valid_loader,
    test_loader=test_loader,
    unique_cells=dataset_valid_test.unique_cellids,
    file_path=FLAGS.output_dir, 
    cells_tensor=dataset_valid_test.unique_cellids_binary,
    label_to_cellid=dataset_valid_test.label_to_cellid,
    best_valid_loss=run_model.best_valid_loss,
    is_single_sample_train=FLAGS.is_single_sample_train
    )
  
  logging.info("Starting to train model.")
  trainer.multi_train_model(model_types)
    

if __name__ == '__main__':
  app.run(main)