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


'''Example command line method to extract Wikipedia and Wikidata items and save 
to file.
Example:
$ bazel-bin/cabby/data/extract_wikigeo_contexts_with_osm \
--region Pittsburgh-small \
--save_path pittsburgh_small.json \
--osm_path  "./cabby/geo/map_processing/poiTestData/pittsburgh_small_poi.pkl"
'''
from absl import logging
from absl import app
from absl import flags
import json
import os

from cabby.data import extract
from cabby.geo import regions


FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "region", None, regions.ALLOWED_REGIONS,
  "Map areas: Manhattan, Pittsburgh or Pittsburgh_small.")
flags.DEFINE_string("save_path", None,
          "The path where the data will be saved.")
flags.DEFINE_string(
  "osm_path", None, 
  "The path where the OSM data will be loaded and used to construct the samples.")


# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("save_path")
flags.mark_flag_as_required("osm_path")


def main(argv):
  del argv  # Unused.

  results = extract.get_data_by_region_with_osm(
    FLAGS.region, FLAGS.osm_path)
  logging.info('The number of results items found is: {}'.format(
    len(results)))

  extract.write_files(FLAGS.save_path, results)

  # Create train, dev, and test sets.
  train_set, dev_set, test_set = extract.split_dataset(results, 0.8, 0.1)

  logging.info('The size of the train-set is: {}'.format(
    len(train_set)))
  logging.info('The size of the dev-set is: {}'.format(
    len(dev_set)))
  logging.info('The size of the test-set is: {}'.format(
    len(test_set)))

  # Create paths for the train, dev, and test sets.
  split_path = FLAGS.save_path.split('.')
  path_without_endidng = split_path[0]
  ending = split_path[-1]

  train_path = path_without_endidng + '_train.' + ending
  dev_path = path_without_endidng + '_dev.' + ending
  test_path = path_without_endidng + '_test.' + ending

  # Write the train, dev, and test sets.
  extract.write_files(train_path, train_set)
  extract.write_files(dev_path, dev_set)
  extract.write_files(test_path, test_set)


if __name__ == '__main__':
  app.run(main)