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

import unittest

import geopandas as gpd
from shapely.geometry.point import Point

from cabby.rvs import templates
from cabby.geo import geo_item

class ObserveTest(unittest.TestCase):

  def testSingleOutput(self):

    all_templates = templates.create_templates()
    picked_template = all_templates.iloc[0]['sentence']

    # Create geo entity .
    geo_landmarks = {}
    geo_features = {}
    geo_landmarks['start_point'] = gpd.GeoSeries(
      {'osmid': 1, 'geometry': Point(), 'main_tag': 'START'})
    geo_landmarks['end_point'] = gpd.GeoSeries(
      {'osmid': 2, 'geometry': Point(), 'main_tag': 'Target Coffee Shop'})
    geo_landmarks['main_pivot'] = gpd.GeoSeries(
      {'osmid': 3, 'geometry': Point(), 'main_tag': 'Food On The Way'})
    geo_landmarks['near_pivot'] = gpd.GeoSeries(
      {'osmid': 4, 'geometry': Point(), 'main_tag': 'Far is Near Travel Agency'})
    geo_landmarks['beyond_pivot'] = gpd.GeoSeries(
      {'osmid': 5, 'geometry': Point(), 'main_tag': 'Beyond The Rainbow Fairy Shop'})
    route = gpd.GeoSeries({'geometry': Point()})
    geo_features['intersections'] = 2
    geo_features['cardinal_direction'] = 'North-North'
    geo_features['spatial_rel_goal'] = 'right'
    geo_features['spatial_rel_pivot'] = 'left'

    entity = geo_item.GeoEntity.add_entity(
      geo_landmarks=geo_landmarks,
      geo_features=geo_features,
      route=route
    )

    instruction, entity_span = templates.add_features_to_template(picked_template, entity)
    expected = "Meet at Target Coffee Shop. Go North-North from Food On The Way for 2 intersections. It will be near Far is Near Travel Agency. If you reach Beyond The Rainbow Fairy Shop, you have gone too far."
    self.assertEqual(instruction, expected)

if __name__ == "__main__":
    unittest.main()
