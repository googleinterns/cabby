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


class WalkTest(unittest.TestCase):

    def setUp(self):
        
        # Load map from disk.
        self.map = map_structure.Map("Manhattan", 18, "/mnt/hackney/data/cabby/poi/")

    def testSingleOutput(self):
        start_point = Point(-73.982473, 40.748432)
        end_point = Point(-73.98403, 40.74907)
        list_points = walk.compute_route(start_point, end_point, self.map.poi_graph, self.map.nodes)

        # Check that two points have arrived.
        self.assertEqual(len(list_points), 2)

        # Check that the correct points have arrived.
        first_point = walk.tuple_from_point(list_points[0])
        second_point = walk.tuple_from_point(list_points[1])
        self.assertEqual(first_point, (40.749102, -73.984076))
        self.assertEqual(second_point, (40.748432, -73.982473))

        # Check that all points are in a bounding box.
        eps = 0.0005
        for point in list_points:
            self.assertLessEqual(point.x, start_point.x + eps)
            self.assertGreaterEqual(point.x, end_point.x - eps)
            self.assertLessEqual(point.y, end_point.y + eps)
            self.assertGreaterEqual(point.y, start_point.y - eps)


if __name__ == "__main__":
    unittest.main()
