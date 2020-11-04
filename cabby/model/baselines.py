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

"""
Baseline models.
RUN - 
$ bazel-bin/cabby/model/text/dual_encoder/model_trainer \
  --data_dir ~/data/RUN  \
  --metrics_dir ~/eval/ \


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

from cabby.model import datasets
from cabby.geo import util as gutil
from cabby.evals import utils as eu
from cabby.model.text import util
from cabby.geo import regions



TASKS = ["RVS", "RUN"]

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
          "The directory from which to load the dataset.")

flags.DEFINE_string("metrics_dir", None,
          "The directory where the metrics evaluation witll be save to.")

flags.DEFINE_enum(
  "task", "RVS", TASKS, 
  "Supported datasets to train\evaluate on: WikiGeo, RVS or RUN.")
          

flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES, 
  regions.REGION_SUPPORT_MESSAGE)

# Required flags.
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("metrics_dir")



def main(argv):
  
  if not os.path.exists(FLAGS.data_dir):
    sys.exit("Dataset path doesn't exist: {}.".format(FLAGS.data_dir))


  metrics_path = os.path.join(FLAGS.metrics_dir, 'metrics.tsv')

  if FLAGS.task == "RUN":
    ds = datasets.RUNDataset(FLAGS.data_dir, 18 )
    ds_test = ds.ds
  else: # RVS
    ds = datasets.RVSDataset(FLAGS.data_dir, 18, FLAGS.region)
    ds_test = ds.test
  
  util.save_metrics_last_only(
      metrics_path, 
      ds_test.end_point.tolist(), 
      ds_test.start_point.tolist())

  logging.info(f"NO-MOVE evaluation for task {FLAGS.task}:")
  evaluator = eu.Evaluator()
  error_distances = evaluator.get_error_distances(metrics_path)
  _, mean_distance, median_distance, max_error, norm_auc = evaluator.compute_metrics(error_distances)


if __name__ == '__main__':
  app.run(main)



