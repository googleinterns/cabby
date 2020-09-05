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
$ bazel-bin/cabby/geo/sample_poi 
--region Manhattan --level 18 --directory "/mnt/hackney/data/cabby/poi/v1/" --path "/mnt/hackney/data/cabby/poi/geo_paths.json" --n_samples 2
'''

from absl import app
from absl import flags

from shapely.geometry.point import Point
import osmnx as ox
from geopandas import GeoDataFrame
import threading

from cabby.geo import walk

from cabby.geo.map_processing import map_structure

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "region", None, ['Pittsburgh', 'Manhattan'],
    "Map areas: Manhattan or Pittsburgh.")
flags.DEFINE_integer("level", None, "Minimum S2 level of the map.")
flags.DEFINE_string("directory", None,
                    "The directory where the map will be loaded from.")
flags.DEFINE_string("path", None,
                    "The path where the files will be saved too.")
flags.DEFINE_integer("n_samples", None, "Number of samples to generate.")


# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("level")
flags.mark_flag_as_required("path")
flags.mark_flag_as_required("n_samples")


def main(argv):
    del argv  # Unused.
    map_region = map_structure.Map(FLAGS.region, FLAGS.level, FLAGS.directory)

    threads = list()
    for index in range(FLAGS.n_samples):
        thread = threading.Thread(target=walk.get_sample, args=(FLAGS.path, map_region))
        threads.append(thread)
        thread.start()




if __name__ == '__main__':
    app.run(main)
