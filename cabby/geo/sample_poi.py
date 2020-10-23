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

'''Command line application to sample an end and starting point, the route  
between them and pivots.
Example:
$ bazel-bin/cabby/geo/sample_poi --region "DC" --min_s2_level 18 \
  --directory "./cabby/geo/map_processing/poiTestData/" \
    --path "./cabby/geo/pathData/dc_geo_paths.gpkg" --n_samples 1
'''

from absl import app
from absl import flags

from geopandas import GeoDataFrame
import multiprocessing 
import osmnx as ox
from shapely.geometry.point import Point

from cabby.geo import regions
from cabby.geo import walk
from cabby.geo.map_processing import map_structure

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)
flags.DEFINE_integer("min_s2_level", None, "Minimum S2 level of the map.")
flags.DEFINE_string("directory", None,
          "The directory where the map will be loaded from.")
flags.DEFINE_string("path", None,
          "The path where the files will be saved to.")
flags.DEFINE_integer("n_samples", None, "Number of samples to generate.")
flags.DEFINE_integer("n_cpu", multiprocessing.cpu_count()-1, "Number of CPUs to use.")


# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("min_s2_level")
flags.mark_flag_as_required("path")
flags.mark_flag_as_required("n_samples")


def main(argv):
  del argv  # Unused.
  map_region = map_structure.Map(
    regions.get_region(FLAGS.region), FLAGS.min_s2_level, FLAGS.directory)

  # Create a file with multiple layers of data.
  walker = walk.Walker(map_region)
  walker.generate_and_save_rvs_routes(FLAGS.path, FLAGS.n_samples, FLAGS.n_cpu)


if __name__ == '__main__':
  app.run(main)