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

import osmnx as ox
import unittest
from shapely.geometry.point import Point

from cabby.geo import walk
from cabby.geo import util
from cabby.geo.map_processing import map_structure


class WalkTest(unittest.TestCase):

  def setUp(self):

    # Load map from disk.
    self.map = map_structure.Map(
      "Bologna", 18)

  def testRouteCalculation(self):
    start_point = Point(11.3414, 44.4951)
    end_point = Point(11.3444, 44.4946)
    route = walk.compute_route(start_point, end_point, self.map.
    nx_graph, self.map.nodes)

    # Check the size of the route. 
    self.assertEqual(route['geometry'].shape[0], 9)

    # Check that the correct points are in the route.
    first_point = util.tuple_from_point(route.iloc[0]['geometry'])
    second_point = util.tuple_from_point(route.iloc[1]['geometry'])
    self.assertEqual(first_point, (44.4946187, 11.344085))
    self.assertEqual(second_point, (44.4947274, 11.343436))


  def testPointsSelection(self):
    geo_entity = walk.get_points_and_route(self.map)
    if geo_entity is None:
      return

    self.assertGreaterEqual(geo_entity.route.shape[0], 1)
    self.assertIsNotNone(geo_entity.end_point['name'])
    self.assertIsNotNone(geo_entity.start_point['name'])
    self.assertIsNotNone(geo_entity.main_pivot['main_tag'])
    self.assertIsNotNone(geo_entity.near_pivot['main_tag'])


if __name__ == "__main__":
  unittest.main()
