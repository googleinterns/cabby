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

'''
Output RVS instructions by templates.

Example command line call:
$ bazel-bin/cabby/rvs/generate_rvs \
  --rvs_data_path /path/to/REGION_geo_paths.gpkg \
  --save_instruction_path /tmp/pittsburgh_instructions.json

Example output: 
  "Meet at Swirl Crepe. Walk past Wellington. Swirl Crepe will be near Gyros."

See cabby/geo/map_processing/README.md for instructions to generate the gpkg 
data file.
'''

import sys

from absl import logging
from absl import app
from absl import flags

from cabby.geo import walk
from cabby.rvs import templates
from cabby.geo import geo_item

FLAGS = flags.FLAGS

flags.DEFINE_string("rvs_data_path", None,
          "The path of the RVS data file to use for generating the RVS instructions.")

flags.DEFINE_string("save_instruction_dir", None,
          "The path of the file where the generated instructions will be saved. ")

flags.DEFINE_float("train_proportion", 0.8,
          "The train proportion of the dataset (0,1)")

flags.DEFINE_float("dev_proportion", 0.1,
          "The dev proportion of the dataset (0,1)")

# Required flags.
flags.mark_flag_as_required('rvs_data_path')
flags.mark_flag_as_required('save_instruction_dir')


def main(argv):
  del argv  # Unused.

  if not FLAGS.train_proportion + FLAGS.dev_proportion < 1:
    sys.exit("Proportion of train and dev combined should be less then 1.")

  logging.info(f"Starting to generate RVS samples")

  entities = walk.load_entities(FLAGS.rvs_data_path)

  if entities is None:
    sys.exit("No entities found.")

  logging.info(f"Number of RVS samples to create: {len(entities)}")

  # Get templates.
  gen_templates = templates.create_templates()

  # Save templates.
  gen_templates['sentence'].to_csv('templates.csv')

  # Split into Train, Dev and Test sets.
  size_templates = gen_templates.shape[0]
  train_size = round(size_templates*FLAGS.train_proportion)
  dev_size = round(size_templates*FLAGS.dev_proportion)

  train_gen_templates = gen_templates[:train_size]
  dev_gen_templates = gen_templates[train_size:train_size+dev_size]
  test_gen_templates = gen_templates[train_size+dev_size:]

  size_entities = len(entities)
  entities_train_size = round(size_entities*FLAGS.train_proportion)
  entities_dev_size = round(size_entities*FLAGS.dev_proportion)

  train_entities = entities[:entities_train_size]
  dev_entities = entities[entities_train_size:entities_train_size+entities_dev_size]
  test_entities = entities[entities_train_size+entities_dev_size:]

  templates.generate_instruction_by_split(train_entities, train_gen_templates, "train", FLAGS.save_instruction_dir)
  templates.generate_instruction_by_split(dev_entities, dev_gen_templates, "dev", FLAGS.save_instruction_dir)
  templates.generate_instruction_by_split(test_entities, test_gen_templates, "test", FLAGS.save_instruction_dir)


if __name__ == '__main__':
  app.run(main)