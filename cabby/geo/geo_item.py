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
'''Basic classes and functions for RVSPath items.'''

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import pandas as pd
import re
from shapely.geometry.point import Point
from shapely.geometry import box, mapping, LineString
from typing import Text, Dict

import attr


@attr.s
class GeoPath:
  """Construct a RVSPath sample.
  `start_point` is the beginning location.
  `end_point` is the goal location.
  `path_features` is a path features: cardinal directions, intersections, instructions and route geometry.
  `main_pivot` is pivot along the route.
  `near_pivot` is the pivot near the goal location.
  `beyond_pivot` is the pivot beyond the goal location.
  """
  start_point: GeoDataFrame = attr.ib()
  end_point: GeoDataFrame = attr.ib()
  path_features: GeoDataFrame = attr.ib()
  main_pivot: GeoDataFrame = attr.ib()
  near_pivot: GeoDataFrame = attr.ib()
  beyond_pivot: GeoDataFrame = attr.ib()
  size: int = attr.ib(init=False)

  def __attrs_post_init__(self):
    self.size = self.beyond_pivot.shape[0]

  @classmethod
  def from_file(cls, start_point, end_point, path_features, main_pivot,
                near_pivot, beyond_pivot):
    """Construct an Entity from the start and end points, route, and pivots.
    """
    return GeoPath(
      start_point,
      end_point,
      path_features,
      main_pivot,
      near_pivot, 
      beyond_pivot
    )

  @classmethod
  def empty(cls):
    """Construct an empty Entity.
    """
    return GeoPath(
      gpd.GeoDataFrame(columns=['osmid', 'geometry', 'main_tag']),
      gpd.GeoDataFrame(columns=['osmid', 'geometry', 'main_tag']),
      gpd.GeoDataFrame(
        columns=[
          'instructions', 
          'geometry',  
          'cardinal_direction',
          'spatial_rel_goal',
          'spatial_rel_pivot',
          'intersections']),
      gpd.GeoDataFrame(columns=['osmid', 'geometry', 'main_tag']),
      gpd.GeoDataFrame(columns=['osmid', 'geometry', 'main_tag']), 
      gpd.GeoDataFrame(columns=['osmid', 'geometry', 'main_tag'])
    )