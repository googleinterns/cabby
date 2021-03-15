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
import random 
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

  # Generate instructions.
  gen_samples = []
  for entity_idx, entity in enumerate(entities):
    current_templates = gen_templates.copy()  # Candidate templates.


    # Use only landmarks that have a main tag.
    for landmark_type, landmark in entity.geo_landmarks.items():
      if landmark_type not in current_templates or landmark_type == "start_point":
        continue
      if not landmark.main_tag:
        current_templates = current_templates[
          current_templates[landmark_type] == False]
      else:
        current_templates = current_templates[current_templates[landmark_type] == True]

    # Use features that exist.
    for feature_type, feature in entity.geo_features.items():

      if feature_type not in current_templates or \
        feature_type=='intersections':
        continue
      if not feature:
        current_templates = current_templates[
          current_templates[feature_type] == False]
      else:
        current_templates = current_templates[current_templates[feature_type] == True]

    intersection = entity.geo_features['intersections']
    intersection = -1 if intersection is None else int(intersection)

    if intersection > 0:
      if intersection == 1:
        # Filter out templates without the next intersection mention.
        current_templates = current_templates[
          current_templates['next intersection'] == True]
      else:
        # Pick templates with either blocks or intersections. X blocks = X intersection - 1.
        # Randomly pick if the feature is a block or an intersection.
        # If 1 - treat it as a block feature, else as an intersection.
        if random.randint(0, 1):
          if intersection == 2:  # one block
            # Filter out templates without the next block mention.
            current_templates = current_templates[
              current_templates['next block'] == True]
          else: # Multiple blocks.
            # Filter out templates without mentions of the number of blocks
            # that should be passed.
            current_templates = current_templates[
              current_templates['blocks'] == True]
        else: # Multiple intersections.
          # Filter out templates without mentions of the number of
          # intersections that should be passed.
          current_templates = current_templates[
            current_templates['intersections'] == True]
    else:
      # Filter out templates with mentions of intersection\block.
      current_templates = current_templates[
        (current_templates['intersections'] == False) &
        (current_templates['blocks'] == False) &
        (current_templates['next intersection'] == False) &
        (current_templates['next block'] == False)]

      # From the candidates left, pick randomly one template.

      choosen_template = current_templates.sample(1)['sentence'].iloc[0]

      gen_instructions, entity_span = templates.add_features_to_template(
        choosen_template, entity)
      rvs_entity = geo_item.RVSSample.to_rvs_sample(
        instructions=gen_instructions,
        id=entity_idx,
        geo_entity=entity,
        entity_span=entity_span
      )
      gen_samples.append(rvs_entity)

  logging.info(f"RVS generated: {len(gen_samples)}")

  uniq_samples = {}
  for gen in gen_samples:
    uniq_samples[gen.instructions] = gen
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
