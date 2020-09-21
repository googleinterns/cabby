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
Example (Come to the Slacker. Go past First Avenue. The Slacker will be near Copies at Carson.):
$ bazel-bin/cabby/rvs/generate_rvs \
	--path "./cabby/geo/pathData/pittsburgh_geo_paths.gpkg" \

'''
from random import randint
from shapely.geometry.point import Point
import sys
from typing import Text

from absl import app
from absl import flags

sys.path.append("/home/tzuf_google_com/dev/cabby")
from cabby.geo import directions
from cabby.geo import util
from cabby.geo import walk
from cabby.rvs import templates

# FLAGS = flags.FLAGS
# flags.DEFINE_string("path", None, "The path of the RVS data file to use for generating the RVS instructions." )


# # Required flags.
# flags.mark_flag_as_required('path')

# entities = walk.read_instructions(FLAGS.path)
entities = walk.read_instructions("/mnt/hackney/data/cabby/pathData/v8/pittsburgh_geo_paths.gpkg")
if entities is None:
  sys.exit("The path to the RVS data was not found.")

# Get templates.
gen_templates = templates.create_templates()

# Generate instructions.
gen_instructions = []
for entity in entities:
  current_templates = gen_templates.copy()
  if entity.tags_start.beyond_pivot is '':
    current_templates = gen_templates[current_templates['beyond_pivot']==False]
  else:
    current_templates = gen_templates[current_templates['beyond_pivot']==True]

  interscrions = -1
  blocks = -1
  if entity.tags_start.intersections > 0:
    is_block = randint(0,1)
    interscrions = -1 if is_block else entity.tags_start.intersections
    blocks = entity.tags_start.intersections-1 if is_block else -1
    if entity.tags_start.intersections == 1:
      current_templates = current_templates[current_templates['next_intersection']==True]
    elif blocks == 1:
      current_templates = current_templates[current_templates['next_block']==True]
    else:
      if blocks >1:
        current_templates = current_templates[current_templates['blocks']==True]
      else:
        current_templates = current_templates[current_templates['intersection']==True]
  else:
    current_templates = current_templates[(current_templates['intersection']==False) & (current_templates['blocks']==False) & (current_templates['next_intersection']==False) & (current_templates['next_block']==False)  ]

  choosen_template = current_templates.sample(1)['sentence'].iloc[0]
  gen_instruction = templates.add_features_to_template(choosen_template, entity.tags_start.end, entity.tags_start.main_pivot, entity.tags_start.near_pivot, entity.tags_start.beyond_pivot, intersection=interscrions, blocks=blocks, cardinal_direction = entity.tags_start.cardina_direction)
  print (gen_instruction)
  gen_instructions.append(gen_instruction)







    







