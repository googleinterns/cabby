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

flags.DEFINE_string("dataset_dir_T5_landmarks_RVS", None,
          "The directory from which to load the dataset.")

flags.DEFINE_string("dataset_dir_T5_landmarks_human", None,
          "The directory from which to load the dataset.")

flags.DEFINE_string("dataset_dir_T5_Warmup_start_end_RVS_fixed_n_4", None,
          "The directory from which to load the dataset.")

flags.DEFINE_string("dataset_dir_T5_Warmup_start_end_RVS_fixed_n_5", None,
          "The directory from which to load the dataset.")

flags.DEFINE_string("dataset_dir_T5_Warmup_Landmarks_NER", None,
          "The directory from which to load the dataset.")
          

flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES, 
  regions.REGION_SUPPORT_MESSAGE)
  


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
  'is_single_sample_train', default=False,
  help=('Train on a single sample and do not evaluate.'))


flags.DEFINE_bool(
  'is_val_loss_from_model', default=False,
  help=('In case the model is loaded - should the validation loss use the models current loss.'))

flags.DEFINE_bool(
  'is_distance_distribution', default=False,
  help=(
    'Add probability over cells according to the distance from start point.'+ 
    'This is optional only for RVS and RUN.'))


# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("s2_level")
flags.mark_flag_as_required("dataset_dir_T5_landmarks_RVS")
flags.mark_flag_as_required("dataset_dir_T5_landmarks_human")
flags.mark_flag_as_required("dataset_dir_T5_Warmup_start_end_RVS_fixed_n_4")
flags.mark_flag_as_required("dataset_dir_T5_Warmup_start_end_RVS_fixed_n_5")
flags.mark_flag_as_required("dataset_dir_T5_Warmup_Landmarks_NER")


def main(argv):
  
  
  dataset_model_path = os.path.join(FLAGS.dataset_dir_T5_landmarks_RVS, "S2-Generation-T5-Landmarks")
  dataset_path = os.path.join(dataset_model_path, str(FLAGS.s2_level))

  unique_cellid_path = os.path.join(dataset_path,"unique_cellid.npy")
  tensor_cellid_path = os.path.join(dataset_path,"tensor_cellid.pth")
  label_to_cellid_path = os.path.join(dataset_path,"label_to_cellid.npy")

  
  path_exists = [
    f for f in [
      FLAGS.dataset_dir_T5_landmarks_RVS, 
      FLAGS.dataset_dir_T5_landmarks_human, 
      FLAGS.dataset_dir_T5_Warmup_start_end_RVS_fixed_n_4,
      FLAGS.dataset_dir_T5_Warmup_start_end_RVS_fixed_n_5] if os.path.isfile(f)]
  
  if not all(path_exists):
    sys.exit()


  dataset_t5_rvs = dataset_item.TextGeoDataset.load(
    dataset_dir = FLAGS.dataset_dir_T5_landmarks_RVS, 
    model_type = "S2-Generation-T5-Landmarks",
    s2_level = FLAGS.s2_level,
    label_to_cellid_path = label_to_cellid_path, 
    unique_cellid_path = unique_cellid_path, 
    tensor_cellid_path = tensor_cellid_path)


  dataset_t5_human = dataset_item.TextGeoDataset.load(
    dataset_dir = FLAGS.dataset_dir_T5_landmarks_human, 
    model_type = "S2-Generation-T5-Landmarks",
    s2_level = FLAGS.s2_level,
    label_to_cellid_path = label_to_cellid_path, 
    unique_cellid_path = unique_cellid_path, 
    tensor_cellid_path = tensor_cellid_path)

  dataset_t5_warmup_4 = dataset_item.TextGeoDataset.load(
    dataset_dir = FLAGS.dataset_dir_T5_Warmup_start_end_RVS_fixed_n_4, 
    model_type = "S2-Generation-T5-Warmup-start-end",
    s2_level = FLAGS.s2_level,
    label_to_cellid_path = label_to_cellid_path, 
    unique_cellid_path = unique_cellid_path, 
    tensor_cellid_path = tensor_cellid_path)

  dataset_t5_warmup_5 = dataset_item.TextGeoDataset.load(
    dataset_dir = FLAGS.dataset_dir_T5_Warmup_start_end_RVS_fixed_n_5, 
    model_type = "S2-Generation-T5-Warmup-start-end",
    s2_level = FLAGS.s2_level,
    label_to_cellid_path = label_to_cellid_path, 
    unique_cellid_path = unique_cellid_path, 
    tensor_cellid_path = tensor_cellid_path)

  dataset_t5_warmup_landmark_ner = dataset_item.TextGeoDataset.load(
    dataset_dir = FLAGS.dataset_dir_T5_Warmup_Landmarks_NER, 
    model_type = "S2-Generation-T5-Warmup-Landmarks-NER",
    s2_level = FLAGS.s2_level,
    label_to_cellid_path = label_to_cellid_path, 
    unique_cellid_path = unique_cellid_path, 
    tensor_cellid_path = tensor_cellid_path)

  train_loader_t5_rvs = DataLoader(
    dataset_t5_rvs.train, batch_size=FLAGS.train_batch_size, shuffle=True)

  train_loader_t5_human = DataLoader(
    dataset_t5_human.train, batch_size=FLAGS.train_batch_size, shuffle=True)

  train_loader_t5_warmup_landmark_ner = DataLoader(
    dataset_t5_warmup_landmark_ner.train, batch_size=FLAGS.train_batch_size, shuffle=True)


  train_loader_t5_warmup_4 = DataLoader(
    dataset_t5_warmup_4.train, batch_size=FLAGS.train_batch_size, shuffle=True)

  train_loader_t5_warmup_5 = DataLoader(
    dataset_t5_warmup_5.train, batch_size=FLAGS.train_batch_size, shuffle=True)


  valid_loader_t5_human = DataLoader(
    dataset_t5_human.valid, batch_size=FLAGS.test_batch_size, shuffle=False)
  test_loader_t5_human = DataLoader(
    dataset_t5_human.test, batch_size=FLAGS.test_batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
  

  run_model = models.S2GenerationModel(
      dataset_t5_rvs.label_to_cellid, is_landmarks=True, is_warmup_start_end=True, device=device)

  run_model.to(device)

  optimizer = torch.optim.Adam(
    run_model.parameters(), lr=FLAGS.learning_rate)
  
  run_model.best_valid_loss = float("Inf")


  trainer = train.Trainer(
    model=run_model,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer,
    train_loader=[
      train_loader_t5_rvs, 
      train_loader_t5_human, 
      train_loader_t5_warmup_4,
      train_loader_t5_warmup_5,
      train_loader_t5_warmup_landmark_ner],
    valid_loader=valid_loader_t5_human,
    test_loader=test_loader_t5_human,
    unique_cells = dataset_t5_human.unique_cellids,
    file_path=FLAGS.output_dir, 
    cells_tensor = dataset_t5_human.unique_cellids_binary,
    label_to_cellid = dataset_t5_human.label_to_cellid,
    is_distance_distribution = FLAGS.is_distance_distribution,
    best_valid_loss = run_model.best_valid_loss,
    is_single_sample_train = FLAGS.is_single_sample_train
    )
  
  logging.info("Starting to train model.")
  trainer.multi_train_model()
    

if __name__ == '__main__':
  app.run(main)