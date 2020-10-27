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


import sys

import networkx as nx
import osmnx as ox
from shapely.geometry.point import Point
from shapely.geometry import box
import unittest

from cabby.geo.map_processing import map_structure
from cabby.geo import regions
from cabby.geo import walk
from cabby.geo import util

class WalkTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):

    # Load map from disk.
    cls.map = map_structure.Map(regions.get_region('DC'), 18)
    cls.walker = walk.Walker(rand_sample = False, map = cls.map)

  def testRouteCalculation(self):
    orig = '#1360050503'
    dest = '#2975872267'
    route = self.walker.compute_route_from_nodes(orig, dest, self.map.
                   nx_graph, self.map.nodes)

    # Check the size of the route.
    self.assertEqual(route['geometry'].shape[0], 26)

    # Check that the correct points are in the route.
    first_point = util.tuple_from_point(route.iloc[0]['geometry'])
    second_point = util.tuple_from_point(route.iloc[1]['geometry'])
    self.assertEqual(first_point, (38.968002, -77.0271067))
    self.assertEqual(second_point, (38.96803286410526, -77.02742946731615))

  def testPointsSelection(self):
    geo_entity = self.walker.get_sample()
    if geo_entity is None:
      return
      
    self.assertIsNotNone(geo_entity)
    self.assertEqual(geo_entity.start_point['osmid'], '#2984603460')
    self.assertEqual(geo_entity.end_point['osmid'], '#1362253177')
    self.assertEqual(geo_entity.main_pivot['osmid'], '#91900570')
    self.assertEqual(geo_entity.near_pivot['osmid'], '#751864718')
    self.assertEqual(geo_entity.intersections, 1)


if __name__ == "__main__":
  unittest.main()
