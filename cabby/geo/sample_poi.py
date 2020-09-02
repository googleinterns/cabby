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

'''Command line application to sample an end and starting point, the route route between them and alandmark on the route.

Example:
$ bazel-bin/cabby/geo/sample_poi \
--region Manhattan --level 18 --directory "/mnt/hackney/data/cabby/poi/v1/"
'''

from absl import app
from absl import flags

from shapely.geometry.point import Point
import osmnx as ox

from cabby.geo import walk
from cabby import logger

from cabby.geo.map_processing import map_structure

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "region", None, ['Pittsburgh', 'Manhattan'],
    "Map areas: Manhattan or Pittsburgh.")
flags.DEFINE_integer("level", None, "Minumum S2 level of the map.")
flags.DEFINE_string("directory", None,
                    "The directory where the files will be saved to")


# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("level")


def main(argv):
    del argv  # Unused.
    map = map_structure.Map(FLAGS.region, FLAGS.level, FLAGS.directory)
    result = walk.get_points_and_route(map)
    while result is None:
        result = walk.get_points_and_route(map)

    end_point, start_point, route, main_pivot, near_pivot = result
    print("Starting at {0} walk past {1} and your goal is {2}, near {3}.".format(start_point['name'], main_pivot['main_tag'], end_point['name'], near_pivot['main_tag']))


if __name__ == '__main__':
    app.run(main)
