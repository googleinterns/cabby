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
from cabby.geo import regions

TASKS = ["WikiGeo", "RVS", "RUN", "human"]

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
          "The directory from which to load the dataset.")
flags.DEFINE_string("dataset_dir", None,
          "The directory to save\load dataloader.")
flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES, 
  regions.REGION_SUPPORT_MESSAGE)
flags.DEFINE_enum(
  "task", "WikiGeo", TASKS, 
  f"Supported datasets to train\evaluate on: {','.join(TASKS)}. ")

flags.DEFINE_enum(
  "model", "Dual-Encoder-Bert", datasets.MODELS, 
  f"Supported models to train\evaluate on:  {','.join(datasets.MODELS)}.")

flags.DEFINE_integer("s2_level", None, "S2 level of the S2Cells.")
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
  'is_distance_distribution', default=False,
  help=(
    'Add probability over cells according to the distance from start point.'+ 
    'This is optional only for RVS and RUN.'))


# Required flags.
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("dataset_dir")
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("s2_level")

def main(argv):
  
  if not os.path.exists(FLAGS.dataset_dir):
    sys.exit("Dataset path doesn't exist: {}.".format(FLAGS.dataset_dir))
  
  dataset_model_path = os.path.join(FLAGS.dataset_dir, str(FLAGS.model))
  dataset_path = os.path.join(dataset_model_path, str(FLAGS.s2_level))
  train_path_dataset = os.path.join(dataset_path,'train.pth')
  valid_path_dataset = os.path.join(dataset_path,'valid.pth')
  test_path_dataset = os.path.join(dataset_path,'test.pth')
  unique_cellid_path = os.path.join(dataset_path,"unique_cellid.npy")
  tensor_cellid_path = os.path.join(dataset_path,"tensor_cellid.pth")
  label_to_cellid_path = os.path.join(dataset_path,"label_to_cellid.npy")


  assert FLAGS.task in TASKS
  if FLAGS.task == "RVS": 
    dataset_init = datasets.RVSDataset
  elif FLAGS.task == 'RUN':
    dataset_init = datasets.RUNDataset
  elif FLAGS.task == 'human':
    dataset_init = datasets.HumanDataset
  else: 
    sys.exit("Dataset invalid")

  dataset = dataset_init(
    data_dir = FLAGS.data_dir, 
    region = FLAGS.region, 
    s2level = FLAGS.s2_level, 
    model_type = FLAGS.model)
 
  if os.path.exists(dataset_path):
    dataset_text = dataset_item.TextGeoDataset.load(
      dataset_path = dataset_path, 
      train_path_dataset = train_path_dataset, 
      valid_path_dataset = valid_path_dataset, 
      test_path_dataset = test_path_dataset, 
      label_to_cellid_path = label_to_cellid_path, 
      unique_cellid_path = unique_cellid_path, 
      tensor_cellid_path = tensor_cellid_path)

  else:
    if not os.path.exists(dataset_model_path):
      os.mkdir(dataset_model_path)
    logging.info("Preparing data.")
    dataset_text = dataset.create_dataset(
            infer_only= FLAGS.infer_only,
    )

    dataset_item.TextGeoDataset.save(
      dataset_text = dataset_text,
      dataset_path = dataset_path, 
      train_path_dataset = train_path_dataset, 
      valid_path_dataset = valid_path_dataset, 
      test_path_dataset = test_path_dataset, 
      label_to_cellid_path = label_to_cellid_path, 
      unique_cellid_path = unique_cellid_path, 
      tensor_cellid_path = tensor_cellid_path)
  
  n_cells = len(dataset_text.unique_cellids)
  logging.info("Number of unique cells: {}".format(
  n_cells))
  train_loader = None
  valid_loader = None
  if FLAGS.infer_only == False:
    train_loader = DataLoader(
      dataset_text.train, batch_size=FLAGS.train_batch_size, shuffle=True)
    valid_loader = DataLoader(
      dataset_text.valid, batch_size=FLAGS.test_batch_size, shuffle=False)
  test_loader = DataLoader(
    dataset_text.test, batch_size=FLAGS.test_batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  logging.info(f"Using model: {FLAGS.model}")
  if 'Dual-Encoder' in FLAGS.model:
    run_model = models.DualEncoder(device=device)
  elif FLAGS.model == 'S2-Generation-T5':
    run_model = models.S2GenerationModel(dataset_text.label_to_cellid)
  elif FLAGS.model == 'S2-Generation-T5-Landmarks':
    run_model = models.S2GenerationModel(dataset_text.label_to_cellid, is_landmarks=True)
  elif FLAGS.model == 'S2-Generation-T5-Path':
    run_model = models.S2GenerationModel(dataset_text.label_to_cellid, is_path=True)
  elif FLAGS.model == 'Classification-Bert':
    run_model = models.ClassificationModel(n_cells)
  else: 
    sys.exit("Model invalid")

    
  if FLAGS.model_path is not None:
    if not os.path.exists(FLAGS.model_path):
      sys.exit(f"The model's path does not exists: {FLAGS.model_path}")
    util.load_checkpoint(
      load_path=FLAGS.model_path, model=run_model, device=device)
  if torch.cuda.device_count() > 1:
    logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
    run_model = nn.DataParallel(run_model).module

  run_model.to(device)

  optimizer = torch.optim.Adam(
    run_model.parameters(), lr=FLAGS.learning_rate)
  
  if FLAGS.is_distance_distribution and FLAGS.task=='WikiGeo':
    sys.exit("Wikigeo does not have a distance distribution option.")

  trainer = train.Trainer(
    model=run_model,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    unique_cells = dataset_text.unique_cellids,
    file_path=FLAGS.output_dir, 
    cells_tensor = dataset_text.unique_cellids_binary,
    label_to_cellid = dataset_text.label_to_cellid,
    is_distance_distribution = FLAGS.is_distance_distribution
    )
  if FLAGS.infer_only:
    logging.info("Starting to infer model.")
    valid_loss, predictions, true_vals, true_points, pred_points = (
      trainer.evaluate(validation_set = False))

    util.save_metrics_last_only(
      trainer.metrics_path, 
      true_points, 
      pred_points)

    accuracy = accuracy_score(true_vals, predictions)

    evaluator = eu.Evaluator()
    error_distances = evaluator.get_error_distances(trainer.metrics_path)
    _, mean_distance, median_distance, max_error, norm_auc = (
      evaluator.compute_metrics(error_distances))

    logging.info(f"\nTest Accuracy: {accuracy}, \n" +
    f"Mean distance: {mean_distance},\nMedian distance: {median_distance},\n" +
    f"Max error: {max_error},\nNorm AUC: {norm_auc}")

  else: 
    logging.info("Starting to train model.")
    trainer.train_model()
    

if __name__ == '__main__':
  app.run(main)