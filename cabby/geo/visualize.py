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

import folium
import geopandas as gpd
from json import JSONDecoder, JSONDecodeError
import json
import pandas as pd
import shapely.geometry as geom
from shapely.geometry import Polygon, Point, LineString
from typing import Tuple, Sequence, Optional, Dict, Text

from cabby.geo import util

def read_file(path: Text) -> gpd.GeoDataFrame:
    '''Read file.
    Arguments:
      path: path to file.
    Returns:
      The GeoDataFrame theat includes the start and end points, pivots and 
      route.
    '''

    start = gpd.read_file(path, layer='start')
    end = gpd.read_file(path, layer='end')
    route = gpd.read_file(path, layer='route')
    main = gpd.read_file(path, layer='main')
    near = gpd.read_file(path, layer='near')

    start = start.rename(columns={'geometry': 'start_point'})
    start['end_point'] = end['geometry']
    start['route'] = route['geometry']
    start['main_pivot_point'] = main['geometry']
    start['near_pivot_point'] = near['geometry']

    return start


def get_osm_map(gdf: gpd.GeoDataFrame) -> Sequence[folium.Map]:
    '''Create the OSM maps.
    Arguments:
      gdf: the GeoDataFrame from which to create the OSM map.
    Returns:
      OSM maps from the GeoDataFrame.
    '''

    mid_point = util.midpoint(gdf.end_point, gdf.start_point)
    zoom_location = util.list_yx_from_point(mid_point)

    # create a map
    map_osm = folium.Map(location=zoom_location,
                         zoom_start=15, tiles='OpenStreetMap')

    # draw the points
    start_point = util.list_yx_from_point(gdf.start_point)
    end_point = util.list_yx_from_point(gdf.end_point)
    main_pivot = util.list_yx_from_point(gdf.main_pivot_point)
    near_pivot = util.list_yx_from_point(gdf.near_pivot_point)

    folium.Marker(start_point, popup='start: ' + gdf.start, \
      icon=folium.Icon(color='pink')).add_to(map_osm)
    folium.Marker(end_point, popup='end: ' + gdf.end, \
      icon=folium.Icon(color='black')).add_to(map_osm)
    folium.Marker(main_pivot, popup='main pivot: ' + \
      gdf.main_pivot, icon=folium.Icon(
        color='orange', icon='info-sign')).add_to(map_osm)
    folium.Marker(near_pivot, popup='near pivot: '+gdf.near_pivot, \
       icon=folium.Icon(
        color='red', icon='info-sign')).add_to(map_osm)

    place_lng, place_lat = gdf.route.exterior.xy

    points = []
    for i in range(len(place_lat)):
        points.append([place_lat[i], place_lng[i]])

    for index, lat in enumerate(place_lat):
        folium.Circle(location = [lat,
                       place_lng[index]],
                       radius = 5,
                      color='crimson',

                      ).add_to(map_osm)

    return map_osm


def get_maps_and_instructions(
  path: Text) -> Tuple[Sequence[folium.Map], Sequence[Text]]:
    '''Create the OSM maps and instructions.
    Arguments:
      path: the path to the .
    Returns:
      OSM maps from the GeoDataFrame.
    '''

    gdf = read_file(path)
    map_osms_instructions = gdf.apply(lambda x: \
      (get_osm_map(x),x.instruction), axis=1)

    return map_osms_instructions

