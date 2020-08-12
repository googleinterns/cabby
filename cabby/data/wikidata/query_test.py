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

'''Tests for wikidata.py'''

from cabby.data.wikipedia import wikidata
import unittest


class WikidataTest(unittest.TestCase):

    def testSingleOutput(self):
        output = wikidata.get_wikidata('Manhattan')
        expected = 'New York Stock Exchange Building'
        poi_by_value_expected = [x for x in output['results']
                                 ['bindings'] if x['placeLabel']['value'] == expected]
        self.assertEqual(len(poi_by_value_expected), 1)


if __name__ == "__main__":
    unittest.main()
