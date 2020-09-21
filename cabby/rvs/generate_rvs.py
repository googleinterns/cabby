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

'''Example command line method to output RVS instructions by templates.
Example (Meet at Swirl Crepe. Walk past Wellington. \
Swirl Crepe will be near Gyros.):
$ bazel-bin/cabby/rvs/generate_rvs \
	--path "./cabby/geo/pathData/pittsburgh_geo_paths.gpkg" \

'''

from random import randint
import sys

from absl import app
from absl import flags

from cabby.geo import walk
from cabby.rvs import templates

FLAGS = flags.FLAGS

flags.DEFINE_string("path", None, 
"The path of the RVS data file to use for generating the RVS instructions." )

# Required flags.
flags.mark_flag_as_required('path')

def main(argv):
  del argv  # Unused.

  entities = walk.read_instructions(FLAGS.path)

  if entities is None:
    sys.exit("The path to the RVS data was not found.")

  print ("Number of RVS samples to create: ",len(entities))

  # Get templates.
  gen_templates = templates.create_templates()

  print ("Number of templates: ",gen_templates.shape[0])

  # Generate instructions.
  gen_instructions = []
  for entity in entities:
    current_templates = gen_templates.copy()
    if entity.tags_start.beyond_pivot is '':
      current_templates = gen_templates[
      current_templates['beyond_pivot']==False]
    else:
      current_templates = gen_templates[current_templates['beyond_pivot']==True]

    if entity.tags_start.intersections > 0:
      is_block = randint(0,1)
      blocks = entity.tags_start.intersections-1 if is_block else -1
      if entity.tags_start.intersections == 1:
        current_templates = current_templates[
        current_templates['next_intersection']==True]
      elif blocks == 1:
        current_templates = current_templates[
        current_templates['next_block']==True]
      else:
        if blocks >1:
          current_templates = current_templates[
          current_templates['blocks']==True]
        else:
          current_templates = current_templates[
          current_templates['intersection']==True]
    else:
      current_templates = current_templates[
      (current_templates['intersection']==False) & 
      (current_templates['blocks']==False) &
      (current_templates['next_intersection']==False) & 
      (current_templates['next_block']==False)  ]

    choosen_template = current_templates.sample(1)['sentence'].iloc[0]
    gen_instruction = templates.add_features_to_template(
    choosen_template, entity)
    gen_instructions.append(gen_instruction)

if __name__ == '__main__':
  app.run(main)






      







