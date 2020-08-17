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

from typing import Text

from geopy.distance import geodesic
from shapely.geometry.point import Point

def get_all_distances(point: Point, entities):
  distances = {}
  for entity in entities:
    distances[entity.qid] = geodesic(point.coords, entity.location.coords)
  return distances


def get_pivot_poi(origin: Point, destination: Point, entities) -> Text:
  pdist = geodesic(origin.coords, destination.coords)
  candidate_scores = {}
  for entity in entities:
    odist = geodesic(origin.coords, entity.location.coords)
    ddist = geodesic(destination.coords, entity.location.coords)
    if odist < pdist and ddist < pdist and odist+ddist < pdist*1.05:
      candidate_scores[entity.qid] = abs(.75 - (odist/(odist+ddist)))

  if candidate_scores:
    ranked_pois = list(candidate_scores.items())
  else:
    distances = get_all_distances(destination.coords, entities)
    ranked_pois = list(distances.items())

  ranked_pois.sort(key=lambda x: x[1])
  return ranked_pois[0][0]
