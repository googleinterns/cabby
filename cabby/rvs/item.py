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

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import pandas as pd
import re
from shapely.geometry.point import Point
from shapely.geometry import box, mapping, LineString
from typing import Text, Dict

import attr


@attr.s
class RVSPath:
  """Construct a RVSPath sample.
  `start_point` is the beginning location.
  `end_point` is the goal location.
  `route` is route between the start and end points.
  `main_pivot` is pivot along the route.
  `near_pivot` is the pivot near the goal location.
  `beyond_pivot` is the pivot beyond the goal location.
  `cardinal_direction` is the cardinal direction betweeen main pivot and 
  goal. Possible cardinal directions: North, South, East, West, North-East, 
  North-West, South-East, South-West. 
  `intersections` is the number of intersections between the pivot along the 
  route and the goal.
  `from_file` should the entity be processed from file.
  `instructions` is a basic template that includes the points, pivots and route 
  features.
  """
  start_point: GeoSeries = attr.ib()
  end_point: GeoSeries = attr.ib()
  route: GeoSeries = attr.ib()
  main_pivot: GeoSeries = attr.ib()
  near_pivot: GeoSeries = attr.ib()
  beyond_pivot: GeoSeries = attr.ib()
  cardinal_direction: Text = attr.ib()
  intersections: int = attr.ib()
  process: bool = attr.ib()
  instructions: Text = attr.ib(init=False)

  def __attrs_post_init__(self):
    if not self.process:
      return
    # Creat basic template instruction.
    if "main_tag" in self.beyond_pivot:
      avoid_instruction = "If you reached {0}, you have gone too far.".format(
        self.beyond_pivot['main_tag'])

    else:
      avoid_instruction = ""
      self.beyond_pivot['main_tag'] = ""

    if self.intersections == 1:
      intersection_instruction = ("and walk straight to the next intersection, "
                    .format(self.intersections))
    elif self.intersections > 1:
      intersection_instruction = ("and walk straight for {0} intersections, "
                    .format(self.intersections))
    else:
      intersection_instruction = ""

    self.instructions = (
      "Starting at {0} walk {1} past {2} {3}and your goal is {4}, near {5}. "
      .format(self.start_point['name'], self.cardinal_direction,
          self.main_pivot['main_tag'], intersection_instruction,
          self.end_point['name'],
          self.near_pivot['main_tag']) + avoid_instruction
    )

    # Create centroid point.
    if self.beyond_pivot['geometry'] is None:
      self.beyond_pivot['geometry'] = Point()

    prune_columns(self.start_point)
    prune_columns(self.end_point)
    prune_columns(self.main_pivot)
    prune_columns(self.near_pivot)
    prune_columns(self.beyond_pivot)

    self.route['cardinal_direction'] = self.cardinal_direction
    self.route['instructions'] = self.instructions
    self.route['intersections'] = self.intersections

  @classmethod
  def from_points_route_pivots(cls, start, end, route, main_pivot,
                              near_pivot, beyond_pivot, cardinal_direction, 
                              intersections):
    """Construct an Entity from the start and end points, route, and pivots.
    """
    return RVSPath(
      start,
      end,
      gpd.GeoSeries(
        {'geometry': LineString(route['geometry'].tolist())}),
      main_pivot,
      near_pivot,
      beyond_pivot,
      cardinal_direction,
      intersections,
      True,
    )

  @classmethod
  def from_file(cls, start, end, route, main_pivot,
                near_pivot, beyond_pivot, cardinal_direction,
                intersections):
    """Construct an Entity from the start and end points, route, and pivots.
    """
    return RVSPath(
      start,
      end,
      route,
      main_pivot,
      near_pivot,
      beyond_pivot,
      cardinal_direction,
      int(intersections),
      False,
    )


def prune_columns(gds: GeoDataFrame):
  """Remove unneeded columns."""
  if 'main_tag' not in gds:
    gds['main_tag'] = gds['name']
  gds.drop(gds.keys().difference(
    ['osmid', 'geometry', 'main_tag']), inplace=True)
