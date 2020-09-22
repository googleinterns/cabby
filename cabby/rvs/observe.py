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
'''Functions to support observation and selection of POIs on paths.'''

from typing import Dict, Sequence, Text

from shapely.geometry.point import Point

from cabby.data.wikidata import item
from cabby.geo import util

# A constant defining how far along we prefer the pivot POI to be between the
# start and the goal.
_IDEAL_PIVOT_DISTANCE_PERCENTAGE = .75

# How much to stretch a path when computing whether the journey from the start
# to a pivot plus from the pivot to the goal is greater than the journey that
# is direct from start to goal.
_PATH_DISTANCE_BUFFER_FACTOR = 1.05

def get_all_distances(
  focus: Point, entities: Sequence[item.WikidataEntity]) -> Dict[Text, float]:
  """Get the distance from each entity to the provided focus point.
  
  Args:
    focus: The point to compute distances to.
    entities: The geolocated entities to compute distances from.
  Returns:
    A dictionary with the QIDs of the entities as keys and the distances from
    the corresponding entity to the focus point.

  """
  distances = {}
  for entity in entities:
    distances[entity.qid] = util.get_distance_km(focus, entity.location)
  return distances
  


def get_pivot_poi(
  start: Point, goal: Point, entities: Sequence[item.WikidataEntity]) -> Text:
  """Select a POI that can act as a useful reference between start and goal.

  Args:
    start: The starting point of a route.
    goal: The goal point of a route.
    entities: The accessible entities to consider.
  Returns:
    The QID of the highest ranked POI linking the start and goal points. 
    Currently this is determined simply by ranking all entities by whether they
    are between the start and goal, and preferring entities that are about 
    three-quarters of the way to the goal from the start. This can be improved
    considerably in future.
  """
  pdist = util.get_distance_km(start, goal)
  candidate_scores = {}
  for entity in entities:
    odist = util.get_distance_km(start, entity.location)
    ddist = util.get_distance_km(goal, entity.location)
    
    if (odist < pdist and
        ddist < pdist and 
        odist+ddist < pdist * _PATH_DISTANCE_BUFFER_FACTOR):
      distance_percentage = odist / (odist + ddist)
      candidate_scores[entity.qid] = abs(
        _IDEAL_PIVOT_DISTANCE_PERCENTAGE - distance_percentage)

  if candidate_scores:
    ranked_pois = list(candidate_scores.items())
  else:
    # No entities are farther from both start and goal than start and goal are
    # from each other. In this case, we rank entities by how close they are to
    # the goal.
    distances = get_all_distances(goal, entities)
    ranked_pois = list(distances.items())

  # Lower scores are better. Sort the POIs, then return the first.
  ranked_pois.sort(key=lambda x: x[1])
  return ranked_pois[0][0]
