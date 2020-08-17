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

Example:
$ bazel-bin/cabby/rvs/generate_rvs \
  --ref_poi "Empire State Building" \
  --goal_poi "pharmacy"
'''

import re
from typing import Text

from absl import app
from absl import flags

import attr
from shapely.geometry.point import Point
import osmnx as ox
import geographiclib
from geopy.distance import geodesic

from cabby.data.wikidata import query
from cabby.geo import directions
from cabby.rvs import speak

FLAGS = flags.FLAGS
flags.DEFINE_string("ref_poi", None, "The reference POI.")
flags.DEFINE_string("goal_poi", None, "The goal POI.")

# Required flags.
flags.mark_flag_as_required("ref_poi")
flags.mark_flag_as_required("goal_poi")

# Ignores situation where multiple points are provided, taking just the first.
_POINT_RE = re.compile(r'^Point\(([-\.0-9]+)\s([-\.0-9]+)\).*$')

@attr.s
class Entity:
  url: Text = attr.ib()
  label: Text = attr.ib()
  location: Point = attr.ib()

  def __attrs_post_init__(self):
    # The QID is the part of the URL that comes after the last / character.
    self.qid = self.url[self.url.rindex('/')+1:]

  @classmethod
  def from_sparql_result(cls, result):
    point_match = _POINT_RE.match(result['point']['value'])
    return Entity(
        result['place']['value'],
        result['placeLabel']['value'],
        Point(float(point_match.group(1)), float(point_match.group(2)))
    )

def get_bearing(start: Point, destination: Point) -> float:
  # Get the bearing (heading) from the start lat-lon to the destination lat-lon.
  # The bearing angle given by azi1 (azimuth) is clockwise relative to north, so
  # a bearing of 90 degrees is due east, 180 is south, and 270 is west. 
  solution = geographiclib.geodesic.Geodesic.WGS84.Inverse(
    start.y, start.x, destination.y, destination.x)
  return directions.angle_in_360(solution['azi1'])

def get_all_distances(point, entities):
  distances = {}
  for entity in entities:
    distances[entity.qid] = geodesic(point.coords, entity.location.coords)
  return distances

def get_pivot_poi(origin, destination, entities):
  pdist = geodesic(origin.coords, destination.coords)
  candidate_scores = {}
  for entity in entities:
    odist = geodesic(origin.coords, entity.location.coords)
    ddist = geodesic(destination.coords, entity.location.coords)
    if odist < pdist and ddist < pdist and odist+ddist < pdist*1.05 :
      candidate_scores[entity.qid] = abs(.75 - (odist/(odist+ddist)))
  
  pivot_relation = ""
  print(candidate_scores)
  if candidate_scores:
    ranked_pois = list(candidate_scores.items())
    pivot_relation = "between"
  else:
    distances = get_all_distances(destination.coords, entities)
    ranked_pois = list(distances.items())
    pivot_relation = "near"
  
  ranked_pois.sort(key=lambda x: x[1])
  return ranked_pois[0][0], pivot_relation

def main(argv):
  del argv  # Unused.

  #print("Obtaining graph for Manhattan.")
  #graph = ox.graph_from_place('Manhattan, New York City, New York, USA')

  #print("Converting the graph to nodes and edge GeoDataFrames.")
  #nodes, _ = ox.graph_to_gdfs(graph)

  start = Point(-73.985085, 40.753628) # Near Bryant Park
  destination = Point(-73.982473, 40.748432)

  path_distance = geodesic(start.coords, destination.coords).km
  bearing = get_bearing(start, destination)
  print(f'Distance {path_distance} km | Bearing {bearing}')
  
  #print(f"Computing route between {start} and {destination}.")
  #route = walk.compute_route(start, destination, graph, nodes)

  #print("Points obtained for the route.")
  #for point in route:
  #  print(point)

  entities = {}
  for result in query.get_geofenced_wikidata_items('Manhattan'):
    entity = Entity.from_sparql_result(result)
    entities[entity.qid] = entity

  # Find the closest POI in the Wikidata items. In future, these will be 
  # sampled from OSM entities.
  distances = get_all_distances(destination, entities.values())
  ranked_pois = list(distances.items())
  ranked_pois.sort(key=lambda x: x[1])
  target_qid, target_distance = ranked_pois[0]
  target_entity = entities[target_qid]
  print(f'Choosing entity {target_entity.label} with QID {target_qid} '
    f'as the destination, which is {target_distance} from the supplied'
    f'destination coordinates {destination}.\n')

  # Remove the destination POI from the entities list so we can identify a
  # pivot POI to use as a reference to get to that destination POI.
  entities.pop(target_qid)
  target_destination = target_entity.location

  pivot_qid, relation = get_pivot_poi(
    start, target_destination, entities.values())
  pivot_entity = entities[pivot_qid]

  start_pivot_bearing = get_bearing(start, pivot_entity.location)
  pivot_dest_bearing = get_bearing(
    pivot_entity.location, target_entity.location)

  print(start_pivot_bearing)
  print(pivot_dest_bearing)
  target_bearing_relative_to_pivot = pivot_dest_bearing - start_pivot_bearing

  pivot_target_dist = geodesic(pivot_entity.location.coords, target_entity.location.coords).km

  direction_description = speak.speak_egocentric_direction(
    directions.get_egocentric_direction(
      target_bearing_relative_to_pivot), pivot_target_dist)
  print(f'{target_entity.label} {direction_description} {pivot_entity.label}')
  

  print(f'Go past {pivot_entity.label} on your way to {target_entity.label}.')

  #print(speak.describe_route(FLAGS.ref_poi, FLAGS.goal_poi))

if __name__ == '__main__':
  app.run(main)
