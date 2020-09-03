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


class WalkTest(unittest.TestCase):

    def setUp(self):
        
        # Load map from disk.
        self.map = map_structure.Map("Bologna", 18)

    def testSingleOutput(self):
        start_point = Point(11.3414, 44.4951)
        end_point = Point(11.3444, 44.4946)
        list_points = walk.compute_route(start_point, end_point, self.map.nx_graph, self.map.nodes)

        # Check the size of the route. 
        self.assertEqual(len(list_points), 9 )

        # Check that the points in the route.
        first_point = walk.tuple_from_point(list_points[0])
        second_point = walk.tuple_from_point(list_points[1])
        self.assertEqual(first_point, (44.4946187, 11.344085))
        self.assertEqual(second_point, (44.4947274, 11.343436))

        # Check that all points are in a bounding box.
        eps = 0.01
        for point in list_points:
            self.assertLessEqual(point.x, start_point.x + eps)
            self.assertGreaterEqual(point.x, end_point.x - eps)
            self.assertLessEqual(point.y, end_point.y + eps)
            self.assertGreaterEqual(point.y, start_point.y - eps)


if __name__ == "__main__":
    unittest.main()
