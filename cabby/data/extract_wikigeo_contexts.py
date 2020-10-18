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
'''Extract and saves Wikigeo items from Wikipedia, Wikidata, and OSM.

Example:
$ bazel-bin/cabby/data/extract_wikigeo_contexts \
  --region "Pittsburgh_small" --output_dir wikigeo

Example with Open Street Map items:
$ bazel-bin/cabby/data/extract_wikigeo_contexts \
  --region "Pittsburgh_small" --output_dir wikigeo \
  --osm_path cabby/geo/map_processing/poiTestData/pittsburgh_small_poi.pkl
'''

from absl import app
from absl import flags
from absl import logging
import json
import os

from cabby.data import extract
from cabby.geo import regions

flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)
flags.DEFINE_string(
  "output_dir", None, "The path where the data will be saved.")
flags.DEFINE_string(
  "osm_path", None,
  "Path to pickled Open Street Map data for the required region.")

# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("output_dir")

FLAGS = flags.FLAGS

def main(argv):
  del argv  # Unused.
  
  logging.info("Extracting Wikigeo samples.")

  # Extract items.
  if FLAGS.osm_path is not None:
    output_prefix = FLAGS.region.lower() + '_osm'
    results = extract.get_data_by_region_with_osm(FLAGS.region, FLAGS.osm_path)
  else:
    output_prefix = FLAGS.region.lower()
    results = extract.get_data_by_region(FLAGS.region)
  logging.info(f'Found {len(results)} items.')
  
  # Split data into train, dev, and test sets.
  splits = extract.split_dataset(results, 0.8, 0.1)

  # Output to disk.
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)


  for split_name, split_data in splits.items():
    logging.info(f'The size of the {split_name} set is {len(split_data)}.')
    output_path = os.path.join(
      FLAGS.output_dir, f'{output_prefix}_{split_name}.json')
    extract.write_files(output_path, split_data)

if __name__ == '__main__':
  app.run(main)
