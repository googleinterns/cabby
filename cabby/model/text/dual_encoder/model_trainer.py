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

"""Dual-encoder framework for text and S2Cellids matching.

Example command line call:
$ bazel-bin/cabby/model/text/dual_encoder/model_trainer \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dual_encoder/dataset/pittsburgh \
  --region Pittsburgh \ 
  --s2_level 12 \
  --output_dir ~/tmp/output/dual\
  --train_batch_size 32 \
  --test_batch_size 32

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
from torch.utils.data import DataLoader
from transformers import AdamW

from cabby.model.text.dual_encoder import train
from cabby.model.text.dual_encoder import dataset
from cabby.model.text.dual_encoder import model
from cabby.model.text.dual_encoder import dataset_item


FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
          "The directory from which to load the dataset.")
flags.DEFINE_string("dataset_dir", None,
          "The directory to save\load dataloader.")
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
  'num_epochs', default=5,
  help=('Number of training epochs.'))


# Required flags.
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("dataset_dir")
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("s2_level")
flags.mark_flag_as_required("output_dir")

def main(argv):

  if not os.path.exists(FLAGS.dataset_dir):
    sys.exit("Dataset path doesn't exist: {}.".format(FLAGS.dataset_dir))

  dataset_path = os.path.join(FLAGS.dataset_dir, str(FLAGS.s2_level))
  train_path_dataset = os.path.join(dataset_path,'train.pth')
  valid_path_dataset = os.path.join(dataset_path,'valid.pth')
  test_path_dataset = os.path.join(dataset_path,'test.pth')
  unique_cellid_path = os.path.join(dataset_path,"unique_cellid.npy")
  tensor_cellid_path = os.path.join(dataset_path,"tensor_cellid.pth")
  label_to_cellid_path = os.path.join(dataset_path,"label_to_cellid.npy")

  if os.path.exists(dataset_path):
    datast_text = dataset_item.TextGeoDataset.load(
      dataset_path = dataset_path, 
      train_path_dataset = train_path_dataset, 
      valid_path_dataset = valid_path_dataset, 
      test_path_dataset = test_path_dataset, 
      label_to_cellid_path = label_to_cellid_path, 
      unique_cellid_path = unique_cellid_path, 
      tensor_cellid_path = tensor_cellid_path)

  else:
    logging.info("Preparing data.")
    datast_text = dataset.create_dataset(
      FLAGS.data_dir, FLAGS.region, FLAGS.s2_level)

    dataset_item.TextGeoDataset.save(
      dataset_text = datast_text,
      dataset_path = dataset_path, 
      train_path_dataset = train_path_dataset, 
      valid_path_dataset = valid_path_dataset, 
      test_path_dataset = test_path_dataset, 
      label_to_cellid_path = label_to_cellid_path, 
      unique_cellid_path = unique_cellid_path, 
      tensor_cellid_path = tensor_cellid_path)
  
  logging.info("Number of unique cells: {}".format(
  len(datast_text.unique_cellids)))

  train_loader = DataLoader(
    datast_text.train, batch_size=FLAGS.train_batch_size, shuffle=True)
  valid_loader = DataLoader(
    datast_text.valid, batch_size=FLAGS.test_batch_size, shuffle=False)
  test_loader = DataLoader(
    datast_text.test, batch_size=FLAGS.test_batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

  dual_encoder = model.DualEncoder()
  if torch.cuda.device_count() > 1:
    logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
    dual_encoder = nn.DataParallel(dual_encoder)

  dual_encoder.to(device)

  optimizer = torch.optim.Adam(dual_encoder.parameters(), lr=FLAGS.learning_rate)
  
  logging.info("Starting to train model.")

  trainer = train.Trainer(
    model=dual_encoder,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    unique_cells = datast_text.unique_cellids,
    file_path=FLAGS.output_dir, 
    cells_tensor = datast_text.unique_cellids_binary,
    label_to_cellid = datast_text.label_to_cellid,
    )
  trainer.train_model()

if __name__ == '__main__':
  app.run(main)



