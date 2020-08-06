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
'''Tests for map_structure.py'''

import map_structure
import osmnx as ox
import unittest
from shapely.geometry.point import Point
from map_structure import Map
from s2geometry import pywraps2 as s2


class MapTest(unittest.TestCase):

    def setUp(self):

        # Process the map for an area in Pittsburgh.
        self.pittsburgh_map = Map("Pittsburgh")

    def testSingleOutput(self):
        # Check a known POI is there.
        self.assertEqual(
            self.pittsburgh_map.poi[self.pittsburgh_map.poi['name'] == 'Frick Building'].shape[0], 1)

        # Check the cellid.
        list_cells = self.pittsburgh_map.poi[self.pittsburgh_map.poi['name'] == 'Frick Building']['cellids'].tolist()[
            0]
        self.assertEqual(list_cells[0].id(), 9814734816715735040)
        self.assertEqual(list_cells[1].id(), 9814734845337665536)
        self.assertEqual(list_cells[2].id(), 9814734856679063552)
        self.assertEqual(list_cells[3].id(), 9814734856712617984)
        self.assertEqual(list_cells[4].id(), 9814734856746172416)
        self.assertEqual(list_cells[5].id(), 9814734856779726848)
        self.assertEqual(list_cells[6].id(), 9814734856813281280)
        self.assertEqual(list_cells[7].id(), 9814734856846835712)

        # Check that the POI was added correctly to the graph.
        cell_to_search = list_cells[0]
        poi = self.pittsburgh_map.graph.search(cell_to_search)
        self.assertTrue(203322568 in poi)


if __name__ == "__main__":
    unittest.main()
