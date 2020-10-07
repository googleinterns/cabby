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
'''Tests for walk.py'''

from cabby.geo.map_processing import map_structure
from cabby.geo import util
from cabby.geo import walk
import osmnx as ox
import unittest
from shapely.geometry.point import Point
import sys


class WalkTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):

    # Load map from disk.
    cls.map = map_structure.Map("DC", 18)

  def testRouteCalculation(self):
    start_point = Point(-77.03994, 38.90842)
    end_point = Point(-77.03958, 38.90830)
    route = walk.compute_route(start_point, end_point, self.map.
                   nx_graph, self.map.nodes)

    # Check the size of the route.
    self.assertEqual(route['geometry'].shape[0], 23)

    # Check that the correct points are in the route.
    first_point = util.tuple_from_point(route.iloc[0]['geometry'])
    second_point = util.tuple_from_point(route.iloc[1]['geometry'])
    self.assertEqual(first_point, (38.908494204353836, -77.03979751967637))
    self.assertEqual(second_point, (38.90910590000001, -77.0395916))

  def testPointsSelection(self):
    geo_entity = walk.get_single_sample(self.map)
    if geo_entity is None:
      return

    self.assertEqual(geo_entity.start_point['osmid'], 4696835894)
    self.assertEqual(geo_entity.end_point['osmid'], 1685776652)
    self.assertEqual(geo_entity.main_pivot['osmid'], 148753535)
    self.assertEqual(geo_entity.near_pivot['osmid'], 1685776628)
    self.assertEqual(geo_entity.beyond_pivot['osmid'], 1704264322)


if __name__ == "__main__":
  unittest.main()