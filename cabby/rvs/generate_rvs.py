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

import json
from random import randint
import sys

from absl import logging
from absl import app
from absl import flags


from cabby.geo import walk
from cabby.rvs import templates
from cabby.rvs import rvs_item


FLAGS = flags.FLAGS

flags.DEFINE_string("rvs_data_path", None,
          "The path of the RVS data file to use for generating the RVS instructions.")

flags.DEFINE_string("save_instruction_path", None,
          "The path of the file where the generated instructions will be saved. ")

# Required flags.
flags.mark_flag_as_required('rvs_data_path')
flags.mark_flag_as_required('save_instruction_path')


def main(argv):
  del argv  # Unused.

  logging.info(f"Starting to generate RVS samples")

  entities = walk.load_entities(FLAGS.rvs_data_path)

  if entities is None:
    sys.exit("No entities found.")

  logging.info(f"Number of RVS samples to create: {len(entities)}")

  # Get templates.
  gen_templates = templates.create_templates()

  logging.info(f"Number of templates: {gen_templates.shape[0]}")

  # Generate instructions.
  gen_samples = []
  for entity_idx, entity in enumerate(entities):
    current_templates = gen_templates.copy()  # Candidate templates.
    if entity.beyond_pivot.main_tag is '':
      # Filter out templates with the beyond pivot mention.
      current_templates = gen_templates[
        current_templates['beyond_pivot'] == False]
    else:
      # Filter out templates without the beyond pivot mention.
      current_templates = gen_templates[current_templates['beyond_pivot'] == True]

    if entity.intersections > 0:
      # Pick templates with either blocks or intersections.
      is_block = randint(0, 1)
      blocks = entity.intersections-1 if is_block else -1
      if entity.intersections == 1:
        # Filter out templates without the next intersection mention.
        current_templates = current_templates[
          current_templates['next_intersection'] == True]
      elif blocks == 1:
        # Filter out templates without the next block mention.
        current_templates = current_templates[
          current_templates['next_block'] == True]
      else:
        if blocks > 1:
          # Filter out templates without mentions of the number of blocks
          # that should be passed.
          current_templates = current_templates[
            current_templates['blocks'] == True]
        else:
          # Filter out templates without mentions of the number of
          # intersections that should be passed.
          current_templates = current_templates[
            current_templates['intersections'] == True]
    else:
      # Filter out templates with mentions of intersection\block.
      current_templates = current_templates[
        (current_templates['intersections'] == False) &
        (current_templates['blocks'] == False) &
        (current_templates['next_intersection'] == False) &
        (current_templates['next_block'] == False)]

    # From the candidates left, pick randomly one template.
    choosen_template = current_templates.sample(1)['sentence'].iloc[0]

    gen_instructions = templates.add_features_to_template(
      choosen_template, entity)
    rvs_entity = rvs_item.RVSData.from_geo_entities(
      start=entity.start_point.geometry,
      start_osmid=entity.start_point.osmid,
      end_osmid=entity.end_point.osmid,
      end=entity.end_point.geometry,
      main_pivot=entity.main_pivot.geometry,
      main_pivot_osmid=entity.main_pivot.osmid,
      near_pivot=entity.near_pivot.geometry,
      near_pivot_osmid=entity.near_pivot.osmid,
      beyond_pivot=entity.beyond_pivot.geometry,
      beyond_pivot_osmid=entity.beyond_pivot.osmid,
      route=entity.route,
      instructions=gen_instructions,
      id=entity_idx,
    )
    gen_samples.append(rvs_entity)

  logging.info(f"RVS generated: {len(gen_samples)}")

  uniq_samples = {}
  for gen in gen_samples:
    uniq_samples[gen.instructions] = gen
  logging.info(f"Unique RVS generated: {len(uniq_samples)}")

  logging.info(
    f"Writing {len(uniq_samples)} samples to file => " +
    f"{FLAGS.save_instruction_path}")
  # Save to file.
  with open(FLAGS.save_instruction_path, 'a') as outfile:
    for sample in uniq_samples.values():
      json.dump(sample, outfile, default=lambda o: o.__dict__)
      outfile.write('\n')
      outfile.flush()

  logging.info("Finished writing to file.")

if __name__ == '__main__':
  app.run(main)
