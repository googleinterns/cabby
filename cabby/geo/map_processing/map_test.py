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

import osmnx as ox
from s2geometry import pywraps2 as s2
from shapely.geometry.point import Point
import unittest

from cabby.geo.map_processing import map_structure


class MapTest(unittest.TestCase):

    def setUp(self):

        # Process the map for an area in Pittsburgh.
        self.pittsburgh_map = map_structure.Map("Pittsburgh", 18)

    def testSingleOutput(self):
        # Verify that a known POI is present.

        specific_poi_found = self.pittsburgh_map.poi[self.pittsburgh_map.poi[
            'name'] == 'Frick Building']
        # Check that the number of Frick Building POI found is exactly 1.
        self.assertEqual(specific_poi_found.shape[0], 1)

        # Check the cellid.
        list_cells = self.pittsburgh_map.poi[self.pittsburgh_map.poi[
            'name'] == 'Frick Building']['cellids'].tolist()[0]
        expected_ids = [
            9814734816715735040, 9814734845337665536, 9814734856679063552,
            9814734856712617984, 9814734856746172416, 9814734856779726848,
            9814734856813281280, 9814734856846835712
        ]
        found_ids = [list_cells[i].id() for i in range(8)]
        for expected, found in zip(expected_ids, found_ids):
            self.assertEqual(expected, found)

        # Check that the POI was added correctly to the graph.
        cell_to_search = list_cells[0]
        node = self.pittsburgh_map.s2_graph.search(cell_to_search)
        self.assertTrue(hasattr(node, 'poi') and 203322568 in node.poi)


if __name__ == "__main__":
    unittest.main()
