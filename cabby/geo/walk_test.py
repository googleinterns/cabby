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
from cabby.geo.map_processing import map_structure
from cabby.geo import util


class WalkTest(unittest.TestCase):

    def setUp(self):

        # Load map from disk.
        self.map = map_structure.Map(
            "Manhattan", 18, "/mnt/hackney/data/cabby/poi/v1/")

    def testRouteCalculation(self):
        start_point = Point(-73.982473, 40.748432)
        end_point = Point(-73.98403, 40.74907)
        route = walk.compute_route(
            start_point, end_point, self.map.nx_graph, self.map.nodes)

        # Check that two points have arrived.
        self.assertEqual(len(route), 2)

        # Check that the correct points 
        first_point = util.tuple_from_point(route.iloc[0]['geometry'])
        second_point = util.tuple_from_point(route.iloc[1]['geometry'])
        self.assertEqual(first_point, (40.749102, -73.984076))
        self.assertEqual(second_point, (40.748432, -73.982473))

        # Check that all points are in a bounding box.
        eps = 0.0005
        for point in route['geometry'].tolist():
            self.assertLessEqual(point.x, start_point.x + eps)
            self.assertGreaterEqual(point.x, end_point.x - eps)
            self.assertLessEqual(point.y, end_point.y + eps)
            self.assertGreaterEqual(point.y, start_point.y - eps)

    def testPointsSelection(self):
        result = walk.get_points_and_route(self.map)
        if result is None:
            return
        print (type(result))
        end_point, start_point, route, main_pivot, near_pivot = result

        self.assertGreaterEqual(route.shape[0], 1)
        self.assertIsNotNone(end_point['name'])
        self.assertIsNotNone(start_point['name'])
        self.assertIsNotNone(main_pivot['main_tag'])
        self.assertIsNotNone(near_pivot['main_tag'])


if __name__ == "__main__":
    unittest.main()
