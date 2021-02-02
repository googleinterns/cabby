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

import geopandas as gpd
from shapely.geometry.point import Point

from cabby.rvs import templates
from cabby.rvs import item

class ObserveTest(unittest.TestCase):

  def testSingleOutput(self):

    all_templates = templates.create_templates()
    picked_template = all_templates.iloc[0]['sentence']

    # Create RVSPath entity.
    start = gpd.GeoSeries({'osmid': 1, 'geometry': Point(), 'main_tag': 'START'})
    end = gpd.GeoSeries({'osmid': 2, 'geometry': Point(), 'main_tag': 'Target Coffee Shop'})
    main = gpd.GeoSeries({'osmid': 3, 'geometry': Point(), 'main_tag': 'Food On The Way'})
    near = gpd.GeoSeries({'osmid': 4, 'geometry': Point(), 'main_tag': 'Far is Near Travel Agency'})
    beyond = gpd.GeoSeries({'osmid': 5, 'geometry': Point(), 'main_tag': 'Beyond The Rainbow Fairy Shop'})
    route = gpd.GeoSeries({'geometry': Point(),
                           'intersections': 2,
                           'cardinal_direction': 'North-North',
                           'spatial_rel_goal': 'right',
                           'spatial_rel_pivot': 'left'
                           })

    entity = item.RVSPath.from_file(
      start,
      end,
      route,
      main,
      near,
      beyond,
      route.cardinal_direction,
      route.spatial_rel_goal,
      route.spatial_rel_pivot,
      route.intersections)

    instruction = templates.add_features_to_template(picked_template,entity)

    expected = "Meet at Target Coffee Shop. Go North-North from Food On The Way for 2 intersections. It will be near Far is Near Travel Agency. If you reach Beyond The Rainbow Fairy Shop, you have gone too far."
    self.assertEqual(instruction, expected)

if __name__ == "__main__":
    unittest.main()
