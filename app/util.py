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
'''Library to support map geographical computations.'''

from collections import namedtuple
from geopy.distance import geodesic
from shapely.geometry.point import Point
from shapely.geometry import LineString
import os
import osmnx as ox
import geopandas as gpd

from typing import Sequence
from typing import List, Optional, Tuple, Sequence, Any, Text

import geo_item

N_AROUND_PIVOTS = 10
N_MAIN_PIVOTS = 15

main_pivots = [f"main_pivot_{n}" for n in range(2, N_MAIN_PIVOTS+1)]
around_pivots = [f"around_goal_pivot_{n}" for n in range(1, N_AROUND_PIVOTS+1)]

LANDMARK_TYPES = [
  "end_point", "start_point", "main_pivot"] + main_pivots + [
    "near_pivot", "beyond_pivot"] + around_pivots

FAR_DISTANCE_THRESHOLD = 2000 # Minimum distance between far cells in meters.
MAX_FAILED_ATTEMPTS = 50

CoordsYX = namedtuple('CoordsYX', ('y x'))
CoordsXY = namedtuple('CoordsXY', ('x y'))



def get_linestring_distance(line: LineString) -> int:
  '''Calculate the line length in meters.
  Arguments:
    route: The line that length calculation will be performed on.
  Returns:
    Line length in meters.
  '''
  dist = 0
  point_1 =  Point(line.coords[0])
  for coord in line.coords[1:]:
    point_2 = Point(coord)
    dist += get_distance_between_points(point_1, point_2)
    point_1 = point_2

  return dist

def get_distance_between_points(
  point_1: Point, point_2: Point) -> int:
  '''Calculate the line length in meters.
  Arguments:
    point_1: The point to calculate the distance from.
    point_2: The point to calculate the distance to.
  Returns:
    Distance length in meters.
  '''

  dist = ox.distance.great_circle_vec(
    point_1.y, point_1.x, point_2.y, point_2.x)

  return dist


def tuple_from_point(point: Point) -> CoordsYX:
  '''Convert a Point into a tuple, with latitude as first element, and
  longitude as second.
  Arguments:
    point(Point): A lat-lng point.
  Returns:
    A lat-lng coordinates.
  '''

  return CoordsYX(point.y, point.x)

def load_entities(path: str):
  if not os.path.exists(path):
    return []
  geo_types_all = {}
  for landmark_type in LANDMARK_TYPES:
    try:
      geo_types_all[landmark_type] = gpd.read_file(path, layer=landmark_type)
    except:
      continue
  geo_types_all['route'] = gpd.read_file(path, layer='path_features')['geometry']
  geo_types_all['path_features'] = gpd.read_file(path, layer='path_features')
  geo_entities = []
  for row_idx in range(geo_types_all[LANDMARK_TYPES[0]].shape[0]):
    landmarks = {}
    for landmark_type in LANDMARK_TYPES:
      if landmark_type in geo_types_all:
        landmarks[landmark_type] = geo_types_all[landmark_type].iloc[row_idx]
    features = geo_types_all['path_features'].iloc[row_idx].to_dict()
    del features['geometry']
    route = geo_types_all['route'].iloc[row_idx]

    geo_item_cur = geo_item.GeoEntity.add_entity(
      geo_landmarks=landmarks,
      geo_features=features,
      route=LineString(route.exterior.coords[:-1])
      )
    geo_entities.append(geo_item_cur)

  return geo_entities

def midpoint(p1: Point, p2: Point) -> Point:
  '''Get the midpoint between two points.
  Arguments:
    p1(Point): A lat-lng point.
    p2(Point): A lat-lng point.
  Returns:
    A lat-lng Point.
  '''
  return Point((p1.x+p2.x)/2, (p1.y+p2.y)/2)

def point_from_str_point(coord_str: Text) -> Point:
  '''Converts coordinates in string format ('Point(longtitude latitude)') to Point.
  E.g, of string 'Point(-74.037258 40.715865)'.
  Arguments:
    coord: A lat-lng coordinate to be converted to a point.
  Returns:
    A point.
  '''
  coords_str = coord_str.replace("POINT (", "").replace(")", "").replace('[','').replace(']','')
  if ',' in coord_str:
    list_coords_str = coords_str.split(',')
  else:
    list_coords_str = coords_str.split(' ')
  coord = list(map(float, list_coords_str))

  return Point(coord[0], coord[1])




def point_from_list_coord_yx(coord: Sequence) -> Point:
  '''Converts coordinates in list format (latitude and longtitude) to Point.
  E.g, of list [40.715865, -74.037258].
  Arguments:
    coord: A lat-lng coordinate to be converted to a point.
  Returns:
    A point.
  '''
  lat = coord[0]
  lon = coord[1]

  return Point(lon, lat)


def point_from_list_coord_xy(coord: Sequence) -> Point:
  '''Converts coordinates in list format (longtitude and latitude) to Point.
  E.g, of list [-74.037258, 40.715865].
  Arguments:
    coord: A lat-lng coordinate to be converted to a point.
  Returns:
    A point.
  '''
  lat = coord[1]
  lon = coord[0]

  return Point(lon, lat)


def list_yx_from_point(point: Point) -> Sequence[float]:
  '''Convert a Point into a sequence, with latitude as first element, and
  longitude as second.
  Arguments:
    point(Point): A lat-lng point.
  Returns:
    A lat-lng Sequence[float, float].
  '''
  return [point.y, point.x]


def get_distance_m(start: Point, goal: Point) -> float:
  """Returns the geodesic distance (in meters) between start and goal.
  This distance is direct (as the bird flies), rather than based on a route
  going over roads and around buildings.
  """
  return geodesic(start.coords, goal.coords).m