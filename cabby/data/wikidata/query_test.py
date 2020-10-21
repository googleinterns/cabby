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

'''Tests for query.py'''

import unittest

from cabby.data.wikidata import query
from cabby.geo import regions

class WikidataTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):

    # Load map from disk.
    cls.manhattan_region = regions.get_region('Manhattan')
    cls.pittsburgh_region = regions.get_region('Pittsburgh')
    cls.items_lean = query.get_geofenced_wikidata_items(cls.manhattan_region)
    cls.items_info = query.get_geofenced_info_wikidata_items(
      cls.manhattan_region)

  def testLeanQuery(self):
    items = self.items_lean
    expected_place_label = 'New York Stock Exchange'
    poi_by_value = [x['placeLabel']['value'] for x in items]
    self.assertIn(expected_place_label, poi_by_value)

    not_expected = 'Rabin Square'

    self.assertNotIn(not_expected, poi_by_value)
    self.assertNotIn('', poi_by_value)

  def testSingleOutputWithoutInstance(self):
    output = query.get_geofenced_wikidata_items(self.pittsburgh_region)
    expected_place_label = 'Arrott Building'
    poi_by_value = [x['placeLabel']['value'] for x in output]
    self.assertIn(expected_place_label, poi_by_value)

  def testRelationsQuery(self):
    wd_relations = query.get_geofenced_wikidata_relations(
        self.pittsburgh_region, extract_qids=True)
    qid_set = set(list(wd_relations.place))
    self.assertIn("Q4915", qid_set)

  def testGetPlaceLocationPointsFromQidNonenglishFail(self):
    # Q6745471 is the Mamaux building in Pittsburgh.
    result = query.get_place_location_points_from_qid(qid="Q6745471")
    self.assertEqual(len(result), 0)

  def testGetPlaceLocationPointsFromQidNonenglishSuccess(self):
    # Q6745471 is the Mamaux building in Pittsburgh.
    result = query.get_place_location_points_from_qid(qid="Q6745471",
                                                      location_only=True)
    self.assertEqual(result[0]['place']['value'], 'http://www.wikidata.org/entity/Q6745471')
    self.assertEqual(result[0]['point']['value'], 'Point(-80.0047 40.4389)')

  def testExtensiveInfoQuery(self):
    items = self.items_info
    place_label = 'New York Stock Exchange'
    print(items)
    poi_by_value = [x for x in items if x['placeLabel']['value']==place_label][0]
    instance_of = poi_by_value['instance']['value']
    self.assertEqual(instance_of, 'stock exchange')
    description = poi_by_value['placeDescription']['value']
    self.assertEqual(description, 'American stock exchange')
  
  def testLeanQueryDuplicates(self):
    # Test for duplicate items.
    items = self.items_lean
    places = [x['place']['value'] for x in items]
    places_set = set(places)
    self.assertEqual(len(places),len(places_set))

if __name__ == "__main__":
  unittest.main()
