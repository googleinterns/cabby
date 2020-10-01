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


from cabby.data.wikidata import query
import unittest


class WikidataTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):

    # Load map from disk.
    cls.items_lean = query.get_geofenced_wikidata_items('Manhattan')
    cls.items_info = query.get_geofenced_info_wikidata_items('Manhattan')

  def testLeanQuery(self):
    items = self.items_lean
    expected_place_label = 'New York Stock Exchange'
    poi_by_value = [x['placeLabel']['value'] for x in items]
    self.assertIn(expected_place_label, poi_by_value)

    not_expected = 'Rabin Square'

    self.assertNotIn(not_expected, poi_by_value)
    self.assertNotIn('', poi_by_value)

  def testExtensiveInfoQuery(self):
    items = self.items_info
    place_label = 'New York Stock Exchange'
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
