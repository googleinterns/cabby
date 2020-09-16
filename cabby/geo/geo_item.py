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
'''Basic classes and functions for Wikigeo items.'''

from geopandas import GeoDataFrame
import re
from shapely.geometry.point import Point
from typing import Text, Dict

import attr


@attr.s
class GeoEntity:
  """Construct a Wikigeo sample.

  `tags_start` includes the tags, instruction and start geolocation.
  `end` is the goal geolocation.
  `route` is geometry of the route between the start and end points.
  `main_pivot` is pivot along the route geolocation.
  `near_pivot` is the pivot near the goal geolocation. 
  `beyond_pivot` is the pivot beyond the goal geolocation. 
  """
  tags_start: GeoDataFrame = attr.ib()
  end: GeoDataFrame = attr.ib()
  route: GeoDataFrame = attr.ib()
  main_pivot: GeoDataFrame = attr.ib()
  near_pivot: GeoDataFrame = attr.ib()
  beyond_pivot: GeoDataFrame = attr.ib()

  @classmethod
  def from_points_route_pivots(cls, start, end, route, main_pivot,
                 near_pivot, beyond_pivot):
    """Construct an Entity from the start and end points, route, and pivots.
    """
    return GeoEntity(
      start,
      end,
      route,
      main_pivot,
      near_pivot,
      beyond_pivot
    )
