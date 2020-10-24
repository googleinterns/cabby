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
    # Test plain Wikipedia query for items.
    items = query.get_wikipedia_items(
      ['New_York_Stock_Exchange', 'Empire_State_Building'])

    # Verify that NYSE lookup gets the right pageid.
    self.assertEqual(items[0].pageid, 21560)

  def testBacklinksItems(self):
    # Test backlinks Wikipedia query for items.
    title = 'Two PNC Plaza'
    items = query.get_backlinks_items_from_wikipedia_titles([title])
    for item in items:
      self.assertIn(title, item.linked_title)

if __name__ == "__main__":
  unittest.main()
