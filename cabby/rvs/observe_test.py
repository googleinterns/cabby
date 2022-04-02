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

'''Tests for speak.py'''

from typing import Dict, Text
import unittest

from shapely.geometry.point import Point

from cabby.data.wikidata import item
from cabby.data.wikidata import query
from cabby.geo import regions
from cabby.rvs import observe


class ObserveTest(unittest.TestCase):

  def testSingleOutput(self):

    # Get Pittsburgh items. Also tests cabby.geo.util.item and 
    # cabby.data.wikidata.query.
    pittsburgh_region = regions.get_region('Pittsburgh')
    pittsburgh_items = [
      item.WikidataEntity.from_sparql_result(result) 
      for result in query.get_geofenced_wikidata_items(pittsburgh_region)
    ]
    pittsburgh_index = {e.qid: e for e in pittsburgh_items}

    # Select five POIs in Pittsburgh.
    market_square = pittsburgh_index['Q6770726']
    warhol_museum = pittsburgh_index['Q751172']
    carnegie_library = pittsburgh_index['Q5043895']
    reserve_bank = pittsburgh_index['Q5440376']
    heinz_hall = pittsburgh_index['Q12059806']

    # Check computed distances from Warhol Museum to the others.
    goal = market_square.location
    pois = [warhol_museum, carnegie_library, reserve_bank, heinz_hall]
    obtained_distances = observe.get_all_distances(goal, pois)
    expected_distances = {
      'Q751172': 0.14883102382530744, 'Q5043895': 0.39191208288190965,
      'Q5440376': 0.8607457546797966, 'Q12059806': 0.09590394273539078
    }

    for qid, expected_distance in expected_distances.items():
      self.assertIn(qid, obtained_distances.keys())
      self.assertAlmostEqual(obtained_distances[qid], expected_distance)

    start = Point(-79.992383, 40.446844) # Near Senator Heinz Center
    obtained_pivot = observe.get_pivot_poi(start, goal, pois)
    self.assertEqual(obtained_pivot, 'Q12059806') # Should be Heinz Hall.

if __name__ == "__main__":
    unittest.main()
