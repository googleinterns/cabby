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

from cabby.data.wikipedia import query
import unittest


class WikipediaTest(unittest.TestCase):

    def testQueryItems(self):
        # Test plain wikipedia query for items
        output = query.get_wikipedia_items(
            ['New_York_Stock_Exchange', 'Empire_State_Building'])
        expected = 'The New York Stock Exchange (NYSE, nicknamed "The Big Board") is an American stock exchange located at 11 Wall Street, Lower Manhattan, New York City'
        self.assertIn(expected, output['21560']['extract'])

        not_expected = 'Rabin Square'

        self.assertNotIn(not_expected, output['21560']['extract'])

    def testBacklinksItems(self):
        # Test backlinks wikipedia query for items
        title = 'New_York_Stock_Exchange'
        entities = query.get_baklinks_items_from_wikipedia_title(title)           
        
        expected = 'New York Stock Exchange'
        not_expected = 'Rabin Square'
        for entity in entities:
            self.assertIn(expected, entity['extract'])
            self.assertNotIn(not_expected, entity['extract'])


if __name__ == "__main__":
    unittest.main()
