# coding=utf-8 Copyright 2020 Google LLC Licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

'''Command line application to output all POI in an area of the OSM.

Example:

$ bazel-bin/cabby/geo/map_processing/map_processor --region "DC" \ 
--min_s2_level 18 --directory "./cabby/geo/map_processing/poiTestData/"
'''

from absl import app
from absl import flags

import osmnx as ox
from shapely.geometry.point import Point

from cabby.geo.map_processing import map_structure

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "region", None, ['Pittsburgh', 'Manhattan','DC'], 
  "Map areas: Manhattan, Pittsburgh or D.C.")
flags.DEFINE_integer("min_s2_level", None, "Minimum S2 level of the map.")

flags.DEFINE_string("directory", None,
          "The directory where the files will be saved to")

# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("min_s2_level")


def main(argv):
  del argv  # Unused.
  map = map_structure.Map(FLAGS.region, FLAGS.min_s2_level)

  if FLAGS.directory is not None:
    # Write to disk.
    map.write_map(FLAGS.directory)

    # Load from disk.
    map_new = map_structure.Map(
      FLAGS.region, FLAGS.min_s2_level, FLAGS.directory)

    print('Number of POI found: {0}'.format(map_new.poi.shape[0]))


if __name__ == '__main__':
  app.run(main)
