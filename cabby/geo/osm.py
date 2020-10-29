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
'''Defines OSM entities adaptations.'''

CORRECTIONS = {'fuel': 'gas station', 'parking': 'parking lot', 'taxi': 'taxi stand', 'doctors': 'doctor\'s office', 'dentist': 'dentist\'s office', 'gambling': 'gambling place', 'police': 'police station', 'shower': 'public bath', 'mall shop': 'mall', 'general shop': 'shop', 'supermarket shop': 'supermarket', 'toys shop': 'toy shop', 'fast food': 'fast food restaurant'}

BLACK_LIST = ['place of worship']

TOURISM_INTRESTING = [ 'aquarium', 'gallery', 'guest_house', 'hostel', 'hotel',
  'motel','museum', 'picnic_site', 'theme_park', 'viewpoint', 'zoo']
           
LEISURE_INTRESTING = ['dog_park', 'fitness_centre', 'garden', 'golfcourse', 
  'ice_rink', 'marina', 'miniature_golf', 'park', 'playground', 'stadium', 
  'water_park']

BUILDING_INTRESTING = ['hotel', 'supermarket', 'warehouse', 'cathedral', 
  'chapel', 'church', 'mosque', 'synagogue', 'hospital', 'kindergarten', 
  'school', 'university', 'train_station', 'stadium', 'bridge']

CRAFT_INTRESTING = ['bakery']

MAN_MADE_INTRESTING = ['tower', 'obelisk', 'bridge', 'lighthouse']

PLACE = ['square']

INTRESTING_TAGS =  {'name': True,
                    'amenity': True,
                    'wikidata': True,
                    'wikipedia': True,
                    'shop': True,
                    'brand': True,
                    'tourism': TOURISM_INTRESTING,
                    'leisure': LEISURE_INTRESTING,
                    'building': BUILDING_INTRESTING,
                    'craft': CRAFT_INTRESTING,
                    'man_made': MAN_MADE_INTRESTING}

NON_SPECIFIC_TAGS = {'amenity': False, 'brand': False, 'shop': 'after', 
  'historic	': 'before', 'tourism': TOURISM_INTRESTING, 'bridge': True,
  'man_made': False, 'natural': False, 'place': PLACE, 
  'leisure': LEISURE_INTRESTING, 'building': BUILDING_INTRESTING, 
  'craft': CRAFT_INTRESTING, 'man_made': MAN_MADE_INTRESTING}


PROMINENT_TAGS_ORDERED = [
  'wikipedia', 'wikidata', 'brand', 'tourism', 'tourism','amenity', 'shop']