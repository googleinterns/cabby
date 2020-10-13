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


from cabby.data import extract
import unittest


class GeoSetTest(unittest.TestCase):


  def testQueryItems(self):
    # Test plain Wikipedia query for items
    two_pnc_plaza_qid = 'Q7859146'
    output = extract.get_data_by_qid(two_pnc_plaza_qid)
    expected_pageid = 12717774  # Two PNC Plaza.
    two_pnc_plazza = [
      x for x in output if x['pageid'] == expected_pageid][0]
    self.assertEqual(two_pnc_plazza['pageid'], expected_pageid)
    self.assertIn('Two PNC Plaza', two_pnc_plazza['text'])
    self.assertEqual(two_pnc_plazza['title'], 'Two PNC Plaza')
    self.assertEqual(two_pnc_plazza['ref_qid'], two_pnc_plaza_qid)
    self.assertEqual(two_pnc_plazza['ref_instance'], 'skyscraper')
  
  def testQueryWithOSM(self):
    samples = extract.get_data_by_region_with_osm("DC")
    self.assertEqual(samples[0]['sample_type'], 'Wikipedia_page')
    self.assertEqual(samples[70]['sample_type'], 'Wikipedia_backlink')
    osm_sample = samples[-2]
    self.assertEqual(osm_sample['sample_type'], 'OSM')
    text_osm = 'Johns Hopkins University of Advanced International Studies and university and Massachusetts Avenue Northwest.'
    self.assertEqual(osm_sample['text'], text_osm)
    wikidata_sample = samples[-5]
    self.assertEqual(wikidata_sample['sample_type'], 'Wikidata')
    text_wikidata = 'Zanvyl Krieger School of Arts and Sciences and academic enviroment and Johns Hopkins University School of Arts and Sciences and academic institution.'
    self.assertEqual(wikidata_sample['text'], text_wikidata)


if __name__ == "__main__":
  unittest.main()
