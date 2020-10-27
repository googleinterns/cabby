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

import sys
sys.path.append("/home/tzuf_google_com/dev/cabby")

import collections
import osmnx as ox
from s2geometry import pywraps2 as s2
from shapely.geometry.point import Point
import unittest

from cabby.geo.map_processing import map_structure
from cabby.geo import regions

class MapTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):

    # Process the map for an area in D.C.
    cls.map = map_structure.Map(regions.get_region('DC'), 18)
  
  def testPOIInGraph(self):
    osmids = self.map.poi['osmid'].tolist()
    for osmid in osmids:
      self.assertIn(osmid, self.map.nx_graph.nodes)

  def testAttributeInGraph(self):
    self.assertIn('1#1360050503', self.map.nx_graph.nodes)
    node = self.map.nx_graph.nodes['1#1360050503']
    self.assertEqual('primary', node['highway'])


  def testSingleOutput(self):
    # Verify that a known POI is present.

    specific_poi_found = self.map.poi[self.map.poi[
      'name'] == 'Dental Dream']
    # Check that the number of Frick Building POI found is exactly 1.
    self.assertEqual(specific_poi_found.shape[0], 1)

    # Check the cellid.
    list_cells = self.map.poi[self.map.poi[
      'name'] == 'Dental Dream']['s2cellids'].tolist()[0]
    expected_ids = [9923620797002285056]
    found_ids = [list_cells[i].id() for i in range(len(list_cells))]
    for expected, found in zip(expected_ids, found_ids):
      self.assertEqual(expected, found)

    

  


if __name__ == "__main__":
  unittest.main()