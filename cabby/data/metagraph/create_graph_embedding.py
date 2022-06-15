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
'''Library to support graph embedding creation from Wikipedia and Wikidata.'''


from absl import app
from absl import flags
from absl import logging

from cabby.geo import regions
from cabby.data.metagraph import utils

flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES, regions.REGION_SUPPORT_MESSAGE)
flags.DEFINE_integer(
    "s2_level", None, "S2 level of the S2Cells.")
flags.DEFINE_multi_integer(
    "s2_node_levels",
    None,
    "Iterable of S2 cell levels (ints) to add to the graph." +
    "The flag can be specified more than once on the command line (the result is a Python list integers).")
flags.DEFINE_string(
    "base_osm_map_filepath", None, "Location of the map_structure.Map to be loaded.")

# Required flags.
flags.mark_flag_as_required("s2_node_levels")
flags.mark_flag_as_required("base_osm_map_filepath")
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("s2_level")

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  utils.construct_metagraph(region=FLAGS.region,
                            s2_level=FLAGS.s2_level,
                            s2_node_levels=FLAGS.s2_node_levels,
                            base_osm_map_filepath=FLAGS.base_osm_map_filepath,
                            )


if __name__ == '__main__':
  app.run(main)
