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

import sys
sys.path.append("/home/tzuf_google_com/dev/cabby")

from cabby.data.wikidata import query
import unittest


class WikidataTest(unittest.TestCase):

  def testSingleOutput(self):
    output = query.get_geofenced_wikidata_items('Manhattan')
    expected_place_label = 'New York Stock Exchange'
    poi_by_value = [x['placeLabel']['value'] for x in output]
    self.assertIn(expected_place_label, poi_by_value)

    not_expected = 'Rabin Square'

    self.assertNotIn(not_expected, poi_by_value)
    self.assertNotIn('', poi_by_value)


if __name__ == "__main__":
  unittest.main()
