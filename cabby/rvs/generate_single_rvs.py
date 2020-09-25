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

'''Example command line method to output simple RVS instructions.
Example (starting near SW corner of Bryant park and heading SE):
$ bazel-bin/cabby/rvs/generate_single_rvs \
  --start_lat 44.49582 --start_lon 11.33697 \
  --goal_lat 44.49268 --goal_lon 11.34365 \
	--region Bologna \
	--min_s2_level 18 \
	--directory "./cabby/geo/map_processing/poiTestData/" \

'''

import sys
from typing import Text

from absl import app
from absl import flags

from shapely.geometry.point import Point

from cabby.geo import directions
from cabby.geo import util
from cabby.geo import walk
from cabby.geo.map_processing import map_structure
from cabby.rvs import item 
from cabby.rvs import observe
from cabby.rvs import speak

FLAGS = flags.FLAGS
flags.DEFINE_float('start_lat', None, 'The latitude of the start.')
flags.DEFINE_float('start_lon', None, 'The longitidue of the start.')
flags.DEFINE_float('goal_lat', None, 'The latitude of the goal.')
flags.DEFINE_float('goal_lon', None, 'The longitidue of the goal.')
flags.DEFINE_enum(
  "region", None, ['Pittsburgh', 'Manhattan','Bologna'], 
  "Map areas: Manhattan, Pittsburgh or Bologna.")
flags.DEFINE_integer("min_s2_level", None, "Minimum S2 level of the map.")
flags.DEFINE_string("directory", None, "The directory where the files will be \
saved to." )


# Required flags.
flags.mark_flag_as_required('start_lat')
flags.mark_flag_as_required('start_lon')
flags.mark_flag_as_required('goal_lat')
flags.mark_flag_as_required('goal_lon')
flags.mark_flag_as_required('region')
flags.mark_flag_as_required('min_s2_level')


def main(argv):
  del argv  # Unused.

  # Example points, to be replaced by points sampled using OSM.
  start = Point(FLAGS.start_lon, FLAGS.start_lat)
  supplied_goal = Point(FLAGS.goal_lon, FLAGS.goal_lat)

  # Get the distance and bearing for the supplied goal location.
  path_distance = util.get_distance_km(start, supplied_goal)
  bearing = util.get_bearing(start, supplied_goal)
  print(f'Distance {path_distance} km | Bearing {bearing}')
  
  # Get all OSM in Manhattan region.
  map = map_structure.Map(FLAGS.region, FLAGS.min_s2_level, FLAGS.directory)

  # Get POIS with names
  pois = map.poi[~map.poi['name'].isnull()]


  # Find the closest POI in the Wikidata items so that we have something to
  # describe. This isn't needed once we use OSM sampled start-goal pairs.
  pois["distance"] = pois.centroid.apply(lambda x: 
	util.get_distance_km(supplied_goal, x))

  target = pois[pois['distance'].min()==pois['distance']].iloc[0]

  print(f'\nChoosing entity {target["name"]} with OSM id {target["osmid"]} '
	f'as the goal, which is {target["distance"]} from the supplied goal '
	f'coordinates {supplied_goal}.\n')

  # Get route. 
  route = walk.compute_route(start, supplied_goal, map.nx_graph, map.nodes)

  # Get the pivots.
  result= walk.get_pivots(route, map, target)
  if result is None:
    sys.exit("No pivots found.")

  main_pivot, near_pivot, beyond_pivot = result

  start_pivot_bearing = util.get_bearing(start, near_pivot["geometry"])
  pivot_dest_bearing = util.get_bearing(
    near_pivot["geometry"], target["centroid"])

  # Computes the bearing of the target by using the bearing from start to
  # pivot as zero. E.g. so instead of 270 meaning west (as usual with bearings),
  # it would mean to the left of the pivot from the perspective of the start.
  target_bearing_relative_to_pivot = pivot_dest_bearing - start_pivot_bearing


  instruction = speak.describe_meeting_point(
    near_pivot["main_tag"], target["name"], target_bearing_relative_to_pivot,
    util.get_distance_km(near_pivot["geometry"], target["centroid"]))

  print(f'Rendezvous instruction:\n\n  {instruction}\n')

if __name__ == '__main__':
  app.run(main)