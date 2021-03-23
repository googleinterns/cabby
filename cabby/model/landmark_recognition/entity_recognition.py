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


"""Entity recognition with Bert encoding.

Example command line call:
$ bazel-bin/cabby/model/landmark_recognition/entity_recognition \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --batch_size 32 \
  --epochs 4 \
  --max_grad_norm 1.0 \
  --region Manhattan
"""

from absl import app
from absl import flags

import os
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, T5ForConditionalGeneration

from cabby.geo import regions
from cabby.model.landmark_recognition import dataset_bert as dataset
from cabby.model.landmark_recognition import run


FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None,
          "The directory from which to load the dataset.")

flags.DEFINE_string("model_path", None,
          "The path to save the model.")

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


# Required flags.
flags.mark_flag_as_required("data_dir")
flags.mark_flag_as_required("model_path")
flags.mark_flag_as_required("batch_size")
flags.mark_flag_as_required("epochs")
flags.mark_flag_as_required("max_grad_norm")
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("s2_level")



def main(argv):
  del argv  # Unused.

  model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
  )

  padSequence = dataset.PadSequence()


  ds_train, ds_val, ds_test = dataset.create_dataset(FLAGS.data_dir, FLAGS.region, FLAGS.s2_level)

  train_dataloader = DataLoader(ds_train, batch_size=FLAGS.batch_size, collate_fn=padSequence)
  val_dataloader = DataLoader(ds_val, batch_size=FLAGS.batch_size, collate_fn=padSequence)
  test_dataloader = DataLoader(ds_test, batch_size=FLAGS.batch_size, collate_fn=padSequence)


  model_trained = run.train(model, train_dataloader, val_dataloader, FLAGS)

  run.test(model_trained, test_dataloader)


if __name__ == '__main__':
  app.run(main)
