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
Example (Head to the GOAL. When you pass MAIN_PIVOT, you'll be just NUMBER_BLOCKS blocks away. The GOAL will be near NEAR_PIVOT.):
$ bazel-bin/cabby/rvs/generate_rvs \
	--path "./cabby/geo/pathData/pittsburgh_geo_paths.gpkg" \

'''

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
entities = walk.read_instructions("/mnt/hackney/data/cabby/pathData/v7/pittsburgh_geo_paths.gpkg")
if entities is None:
  sys.exit("The path to the RVS data was not found.")

# Get templates.
templates = templates.create_templates()

for entity in entities:
  print(entity)
  if entity.beyond_pivot is None:
    current_templates = templates[templates['beyond_pivot']==False]
  else:
    current_templates = templates[templates['beyond_pivot']==True]
  # if entity.tags_start





