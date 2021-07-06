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


"""Pivot recognition with T5 encoding.

Example command line call:
$ bazel-bin/cabby/model/landmark_recognition/entity_recognition \
  --data_dir ~/data/RVS/Manhattan  \
  --batch_size 32 \
  --epochs 4 \
  --max_grad_norm 1.0 \
  --region Manhattan
  --data_dir_touchdown ~/data/Touchdown/
  --data_dir_run ~/data/RUN/
  --n_samples 5
  --pivot_type end_pivot
"""

from absl import app
from absl import flags

import enum
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration

from cabby.geo import regions
from cabby.model.landmark_recognition import dataset_t5 as dataset
from cabby.model.landmark_recognition import run_t5
from cabby.geo import walk

FLAGS = flags.FLAGS

landmark_list = walk.LANDMARK_TYPES + [dataset.EXTRACT_ALL_PIVOTS]
landmark_msg = "Landmark type: " + ','.join(landmark_list)

flags.DEFINE_string("data_dir", None,
          "The directory from which to load the dataset.")

flags.DEFINE_string("model_prefix", None,
          "The path to save the model.")

flags.DEFINE_enum(
  "pivot_type", None, landmark_list,
  landmark_msg)

flags.DEFINE_integer(
  'batch_size', default=32,
  help=('Batch size.'))

flags.DEFINE_integer(
  'epochs', default=4,
  help=('Epochs size.'))

flags.DEFINE_float(
  'max_grad_norm', default=1.0,
  help=('Max grad norm.'))

flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)

flags.DEFINE_integer(
  "s2_level", default=18, help=("S2 level of the S2Cells."))


flags.DEFINE_integer(
  "n_samples", default=5, help=("Number of samples to test on."))



flags.DEFINE_string("data_dir_touchdown", None,
          "The directory from which to load the Touchdown dataset for testing.")

flags.DEFINE_string("data_dir_run", None,
          "The directory from which to load the RUN dataset for testing.")


# Required flags.
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("model_prefix")
flags.mark_flag_as_required("pivot_type")


def main(argv):
  del argv  # Unused.

  train_config = FLAGS.flag_values_dict()
  train_config['model_path'] = FLAGS.model_prefix + "_" + FLAGS.pivot_type + ".pt"
  train_config = type('Config', (object,), train_config)

  model = T5ForConditionalGeneration.from_pretrained(
    "t5-small",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
  )

  ds_train, ds_val, ds_test = dataset.create_dataset(
    FLAGS.data_dir, FLAGS.region, FLAGS.s2_level, FLAGS.pivot_type)

  train_dataloader = DataLoader(
    ds_train, batch_size=FLAGS.batch_size)
  val_dataloader = DataLoader(ds_val, batch_size=FLAGS.batch_size)
  test_dataloader = DataLoader(ds_test, batch_size=FLAGS.batch_size)

  model_trained = run_t5.train(model, train_dataloader, val_dataloader, train_config)

  run_t5.test(model_trained, test_dataloader)

  print ("\n Samples from RVS:")
  instructions = ds_test.ds.instructions.sample(FLAGS.n_samples).tolist()
  run_t5.test_samples(
    instructions=instructions,
    tokenizer=dataset.tokenizer,
    model=model_trained
  )

  print ("\n Samples from RUN:")
  if os.path.exists(FLAGS.data_dir_run):
    path_run = os.path.join(FLAGS.data_dir_run, 'dataset.json')
    if os.path.exists(path_run):
      run_ds = pd.read_json(path_run, lines=True)
      instructions = run_ds.instruction.sample(FLAGS.n_samples).tolist()
      run_t5.test_samples(
        instructions=instructions,
        tokenizer=dataset.tokenizer,
        model=model_trained
      )

  print ("\n Samples from Touchdown:")
  # Test against Touchdown dataset (no labels for landmarks).
  if os.path.exists(FLAGS.data_dir_touchdown):
    path_touchdown = os.path.join(FLAGS.data_dir_touchdown, 'test.json')
    if os.path.exists(path_touchdown):
      touchdown_ds = pd.read_json(path_touchdown, lines=True)
      instructions = touchdown_ds.navigation_text.sample(FLAGS.n_samples).tolist()
      run_t5.test_samples(
        instructions=instructions,
        tokenizer=dataset.tokenizer,
        model=model_trained
      )


if __name__ == '__main__':
  app.run(main)
