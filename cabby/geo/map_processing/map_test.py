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

  @classmethod
  def setUpClass(self):
  
    # Process the map for an area in Bologna.
    self.bologna_map = map_structure.Map("Bologna", 18)

  def testSingleOutput(self):
    # Verify that a known POI is present.

    specific_poi_found = self.bologna_map.poi[self.bologna_map.poi[
      'name'] == 'Delizie di Forno']
    # Check that the number of Frick Building POI found is exactly 1.
    self.assertEqual(specific_poi_found.shape[0], 1)

    # Check the cellid.
    list_cells = self.bologna_map.poi[self.bologna_map.poi[
      'name'] == 'Delizie di Forno']['cellids'].tolist()[0]
    expected_ids = [5152070235402010624]
    found_ids = [list_cells[i].id() for i in range(len(list_cells))]
    for expected, found in zip(expected_ids, found_ids):
      self.assertEqual(expected, found)

    # Check that the POI was added correctly to the graph.
    cell_to_search = list_cells[0]
    node = self.bologna_map.s2_graph.search(cell_to_search)
    self.assertTrue(hasattr(node, 'poi') and 4696883190 in node.poi)


if __name__ == "__main__":
  unittest.main()
