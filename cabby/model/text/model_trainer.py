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
from transformers import AdamW, Adafactor

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
  "train_region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)

flags.DEFINE_enum(
  "dev_region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)

flags.DEFINE_enum(
  "test_region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)

flags.DEFINE_enum(
  "task", "WikiGeo", TASKS,
  f"Supported datasets to train\evaluate on: {','.join(TASKS)}. ")

flags.DEFINE_enum(
  "model", "Dual-Encoder-Bert", dataset_item.MODELS,
  f"Supported models to train\evaluate on:  {','.join(dataset_item.MODELS)}.")

flags.DEFINE_integer("s2_level", None, "S2 level of the S2Cells.")

flags.DEFINE_string("output_dir", None,
                    "The directory where the model and results will be save to.")

flags.DEFINE_float(
  'learning_rate', default=1e-4,
  help=('The learning rate for the Adam optimizer.'))

flags.DEFINE_string("model_path", None,
                    "A path of a model the model to be fine tuned\ evaluated.")

flags.DEFINE_string("train_graph_embed_path", default="",
                    help="The path to the graph embedding.")

flags.DEFINE_string("dev_graph_embed_path", default="",
                    help="The path to the graph embedding.")

flags.DEFINE_string("test_graph_embed_path", default="",
                    help="The path to the graph embedding.")

flags.DEFINE_integer(
  'train_batch_size', default=4,
  help=('Batch size for training.'))

flags.DEFINE_integer(
  'test_batch_size', default=4,
  help=('Batch size for testing and validating.'))

flags.DEFINE_integer(
  'num_epochs', default=5,
  help=('Number of training epochs.'))

flags.DEFINE_integer(
  'n_fixed_points', default=4,
  help=('In case model is S2-Generation-T5-Warmup-start-end pick number of fixed points to generate.'))

flags.DEFINE_integer(
  'far_distance_threshold', default=2000,
  help=('Used when sampling far cells.' +
        'A far cell is defined be a minimum distance (in meters) from a certain cell.'))

flags.DEFINE_bool(
  'infer_only', default=False,
  help=('Train and infer\ just infer.'))

flags.DEFINE_bool(
  'is_single_sample_train', default=False,
  help=('Train on a single sample and do not evaluate.'))

flags.DEFINE_bool(
  'is_val_loss_from_model', default=False,
  help=('In case the model is loaded - should the validation loss use the models current loss.'))

flags.DEFINE_bool(
  'is_distance_distribution', default=False,
  help=(
    'Add probability over cells according to the distance from start point.' +
    'This is optional only for RVS and RUN.'))

flags.DEFINE_integer(
  'graph_codebook', default=0,
  help=('graph quantization'))


# Required flags.
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("dataset_dir")
flags.mark_flag_as_required("train_region")
flags.mark_flag_as_required("dev_region")
flags.mark_flag_as_required("test_region")
flags.mark_flag_as_required("s2_level")


def main(argv):

  logging.info('parameter values:')
  for name in FLAGS:
    logging.info(f'  {name}: {getattr(FLAGS, name)}')
    
  if not os.path.exists(FLAGS.dataset_dir):
    sys.exit("Dataset path doesn't exist: {}.".format(FLAGS.dataset_dir))

  dataset_model_path = os.path.join(FLAGS.dataset_dir, str(FLAGS.model))
  dataset_path = os.path.join(dataset_model_path, str(FLAGS.s2_level))



  assert FLAGS.task in TASKS
  if FLAGS.task == "RVS":
    dataset_init = datasets.RVSDataset
  elif FLAGS.task == 'RUN':
    dataset_init = datasets.RUNDataset
  elif FLAGS.task == 'human':
    dataset_init = datasets.HumanDataset
  else:
    sys.exit("Dataset invalid")

  if FLAGS.is_single_sample_train:
    FLAGS.train_batch_size = 1


  if os.path.exists(dataset_path):
    dataset_text = dataset_item.TextGeoDataset.load(
      dataset_dir=FLAGS.dataset_dir,
      model_type=str(FLAGS.model),
      s2_level=FLAGS.s2_level,
      train_region=FLAGS.train_region,
      dev_region=FLAGS.dev_region,
      test_region=FLAGS.test_region 
    )

  else:
    dataset = dataset_init(
      data_dir=FLAGS.data_dir,
      train_region=FLAGS.train_region,
      dev_region=FLAGS.dev_region,
      test_region=FLAGS.test_region,
      s2level=FLAGS.s2_level,
      model_type=FLAGS.model,
      n_fixed_points=FLAGS.n_fixed_points,
      train_graph_embed_path=FLAGS.train_graph_embed_path,
      dev_graph_embed_path=FLAGS.dev_graph_embed_path,
      test_graph_embed_path=FLAGS.test_graph_embed_path)

    if not os.path.exists(dataset_model_path):
      os.mkdir(dataset_model_path)
    logging.info("Preparing data.")
    dataset_text = dataset.create_dataset(
      infer_only=FLAGS.infer_only,
      is_dist=FLAGS.is_distance_distribution,
      far_cell_dist=FLAGS.far_distance_threshold
    )

    dataset_item.TextGeoDataset.save(dataset_text=dataset_text, dataset_dir=dataset_path)

  train_loader = None
  valid_loader = None
  if FLAGS.infer_only == False:
    train_loader = DataLoader(
      dataset_text.train_set, batch_size=FLAGS.train_batch_size, shuffle=True)
    valid_loader = DataLoader(
      dataset_text.dev_set, batch_size=FLAGS.test_batch_size, shuffle=False)
  test_loader = DataLoader(
    dataset_text.test_set, batch_size=FLAGS.test_batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

  logging.info(f"Using model: {FLAGS.model}")
  if 'Dual-Encoder' in FLAGS.model:
    run_model = models.DualEncoder(
      device=device, is_distance_distribution=FLAGS.is_distance_distribution)
  elif 'T5' in FLAGS.model:
    run_model = models.S2GenerationModel(
      label_to_cellid=dataset_text.test_set.coord_to_cellid, device=device, model_type=FLAGS.model,
      vq_dim=dataset_text.train_set.graph_embed_size,
      graph_codebook=FLAGS.graph_codebook)
  elif FLAGS.model == 'Classification-Bert':
    n_cells = len(dataset_text.test_set.unique_cellids)
    run_model = models.ClassificationModel(n_cells, device=device)
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


  optimizer = AdamW(
    run_model.parameters(), weight_decay=0.001, lr=FLAGS.learning_rate)

  if FLAGS.is_distance_distribution and FLAGS.task == 'WikiGeo':
    sys.exit("Wikigeo does not have a distance distribution option.")

  if not FLAGS.is_val_loss_from_model:
    run_model.best_valid_loss = float("Inf")
  else:
    logging.info(f"Current validation loss: {run_model.best_valid_loss}")


  trainer = train.Trainer(
    model=run_model,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    unique_cells=dataset_text.train_set.unique_cellids,
    file_path=FLAGS.output_dir,
    cells_tensor_dev=dataset_text.dev_set.unique_cellids_binary,
    cells_tensor_test=dataset_text.test_set.unique_cellids_binary,
    label_to_cellid_dev=dataset_text.dev_set.coord_to_cellid,
    label_to_cellid_test=dataset_text.test_set.coord_to_cellid,
    best_valid_loss=run_model.best_valid_loss,
    is_single_sample_train=FLAGS.is_single_sample_train
  )
  if FLAGS.infer_only:
    logging.info("Starting to infer model.")
    valid_loss, predictions, true_vals, true_points, pred_points, start_points = (
      trainer.evaluate(validation_set=False))

    util.save_metrics_last_only(
      trainer.metrics_path,
      true_points,
      pred_points,
      start_points)

    evaluator = eu.Evaluator()
    error_distances = evaluator.get_error_distances(trainer.metrics_path)
    evaluator.compute_metrics(error_distances)

  else:
    logging.info("Starting to train model.")
    trainer.train_model()


if __name__ == '__main__':
  app.run(main)