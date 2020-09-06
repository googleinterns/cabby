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

'''Library to support geographical visualization.'''

from typing import Tuple, Sequence, Optional, Dict, Text
from json import JSONDecoder, JSONDecodeError
import json
import pandas as pd
import folium

import util
import item


def read_file(path: Text):
    json_file = open(path)
    entities = []
    for line in json_file:
        json_line = json.loads(line)
        entity = item.GeoEntity.from_api_result(json_line)
        entities.append(entity)
    return entities


def get_osm_map(entity):

    mid_point = util.midpoint(entity.end_point,entity.start_point)
    zoom_location= util.list_yx_from_point(mid_point)

    # create a map
    map_osm = folium.Map(location=zoom_location, zoom_start=15, tiles='OpenStreetMap') 
    style_function = lambda x: {'weight': 1, 'fillColor':'#eea500'}
        
    # draw the points
    start_point= util.list_yx_from_point(entity.start_point)
    end_point= util.list_yx_from_point(entity.end_point)
    pivot= util.list_yx_from_point(entity.pivot)
    
    folium.Marker(start_point, popup='start point').add_to(map_osm)
    folium.Marker(end_point, popup='end point').add_to(map_osm)
    folium.Marker(pivot, popup='pivot', icon=folium.Icon(color='red', icon='info-sign')).add_to(map_osm)

    return map_osm

def get_maps_and_instructions(path: Text):
    entities = read_file(path)
    osm_maps=[]
    for entity in entities:
      osm_maps.append((get_osm_map(entity),entity.instruction))
    return osm_maps


