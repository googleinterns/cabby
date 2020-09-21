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
class RVSPath:
  """Construct a Wikigeo sample.
  `start_point` is the beginning location.
  `end_point` is the goal location.
  `route` is route between the start and end points.
  `main_pivot` is pivot along the route.
  `near_pivot` is the pivot near the goal location.
  `beyond_pivot` is the pivot beyond the goal.
  `cardinal_direction` is the cardinal direction between the main pivot and the 
  goal. Possible cardinal directions: North, South, East, West, North-East, 
  North-West, South-East, South-West. 
  `instruction` is a basic template that includes the points and pivots.
  """
  start_point: Dict = attr.ib()
  end_point: Dict = attr.ib()
  route: GeoDataFrame = attr.ib()
  main_pivot: Dict = attr.ib()
  near_pivot: Dict = attr.ib()
  beyond_pivot: Dict = attr.ib()
  cardinal_direction: Text = attr.ib()
  instruction: Text = attr.ib(init=False)

  def __attrs_post_init__(self):

    # Creat basic template instruction.
    if "main_tag" in self.beyond_pivot:
      avoid_instruction = "If you reached {0}, you have gone too far.".format(
        self.beyond_pivot['main_tag'])
    else:
      avoid_instruction = ""

    self.instruction = \
      "Starting at {0} walk {1} past {2} and your goal is {3}, near {4}. " \
      .format(self.start_point['name'], self.cardinal_direction, self.
          main_pivot['main_tag'], self.end_point['name'],
          self.near_pivot['main_tag']) + avoid_instruction

    # Creat basic template instruction.
    if "main_tag" in self.beyond_pivot:
      avoid_instruction = "If you reached {0}, you have gone too far.".format(
      self.beyond_pivot['main_tag'])
    else:
      avoid_instruction = ""

    self.instruction = (
      "Starting at {0} walk past {1} and your goal is {2}, near {3}. "
      .format(self.start_point['name'], self.main_pivot['main_tag'], self.
        end_point['name'], self.near_pivot['main_tag']) +
        avoid_instruction
    )

    # Create centroid point.
    if self.beyond_pivot['geometry'] is None:
      self.beyond_pivot['centroid'] = Point()
    if "main_tag" in self.beyond_pivot:
      self.beyond_pivot['main_tag']=self.beyond_pivot['main_tag']
    else:
      self.beyond_pivot['main_tag']= ""

      self.beyond_pivot['main_tag'] = self.beyond_pivot['main_tag'] if "main_tag" in self.beyond_pivot else ""

  @classmethod
  def from_points_route_pivots(cls, start, end, route, main_pivot,
                 near_pivot, beyond_pivot, cardinal_direction):
    """Construct an Entity from the start and end points, route, and pivots.
    """
    return RVSPath(
      start,
      end,
      route,
      main_pivot,
      near_pivot,
      beyond_pivot,
      cardinal_direction
    )
