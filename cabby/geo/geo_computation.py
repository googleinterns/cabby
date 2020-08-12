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

'''Example command line method to compute the route between two given points.'''

from absl import app
from absl import flags

from shapely.geometry.point import Point
import osmnx as ox
from geo import walk


FLAGS = flags.FLAGS
flags.DEFINE_float("orig_lat", None, "origin latitude.")
flags.DEFINE_float("orig_lon", None, "origin longtitude.")
flags.DEFINE_float("dest_lat", None, "destination latitude.")
flags.DEFINE_float("dest_lon", None, "destination longtitude.")

# Required flags.
flags.mark_flag_as_required("orig_lat")
flags.mark_flag_as_required("orig_lon")
flags.mark_flag_as_required("dest_lat")
flags.mark_flag_as_required("dest_lon")


def main(argv):
    del argv  # Unused.

    # Compute graph over Manhattan.
    graph = ox.graph_from_place('Manhattan, New York City, New York, USA')

    # Convert a graph to nodes and edge GeoDataFrames.
    nodes, _ = ox.graph_to_gdfs(graph)

    print(
        walk.compute_route(
            Point(FLAGS.orig_lon, FLAGS.orig_lat),
            Point(FLAGS.dest_lon, FLAGS.dest_lat), graph, nodes))


if __name__ == '__main__':
    app.run(main)
