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
import folium
import geographiclib
from geopy.distance import geodesic
import numpy as np
from numpy import int64
import operator
import osmnx as ox
import pandas as pd
from s2geometry import pywraps2 as s2
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry import box, mapping, LineString
import sys
from typing import Optional, Tuple, Sequence, Any, Text
import webbrowser

FAR_DISTANCE_THRESHOLD = 2000 # Minimum distance between far cells in meters.
MAX_FAILED_ATTEMPTS = 50

CoordsYX = namedtuple('CoordsYX', ('y x'))
CoordsXY = namedtuple('CoordsXY', ('x y'))

def get_distance_between_points(start_point: Point, end_point: Point) -> float:
  '''Calculate the line length in meters.
  Arguments:
    start_point: The point to calculate the distance from.
    end_point: The point to calculate the distance to.
  Returns:
    Distance length in meters.
  '''

  return ox.distance.great_circle_vec(
    start_point.y, start_point.x, end_point.y, end_point.x)

def far_cellid(point: Point, cells: pd.DataFrame) -> Optional[float]:
  '''Get a cell id far from the given cell point. 
  Arguments:
    point: The center point of the cell.
  Returns:
    A cellid of a far cell.
  '''
  far_cell_found = None
  failed_counter = 0
  while far_cell_found is None:
    failed_counter += 1
    if failed_counter > MAX_FAILED_ATTEMPTS:
      sys.exit(
        f"Reached max number of failed attempts in far cell calculation for point: {
        (Point.y, Point.x)} from {(sample_cell.point.y, sample_cell.point.x)}.")
    sample_cell = cells.sample(1).iloc[0]
    distance = get_distance_between_points(point, sample_cell.point) 
    if distance > FAR_DISTANCE_THRESHOLD:
      far_cell_found = sample_cell.cellid
    
    

  return far_cell_found

def neighbor_cellid(cellid: int) -> int:
  '''Get a neighbor cell id. 
  Arguments:
    cellid: The cellid of the cell to return a neighbor cellid for.
  Returns:
    A cellid of a neighbor cell.
  '''

  cell = s2.S2CellId(cellid)
  return cell.next().id()

def cellids_from_s2cellids(list_s2cells: Sequence[s2.S2CellId]) -> Sequence[int]:
  '''Converts a sequence of S2CellIds to a sequence of ids of the S2CellIds. 
  Arguments:
    list_s2cells(S2CellIds): The list of S2CellIds to be converted to ids.
  Returns:
    A sequence of ids corresponding to the S2CellIds.
  '''

  return [cell.id() for cell in list_s2cells]


def s2cellids_from_cellids(list_ids: Sequence[int]) -> Sequence[s2.S2CellId]:
  '''Converts a sequence of ids of S2CellIds to a sequence of S2CellIds. 
  Arguments:
    list_ids(list): The list of S2CellIds ids to be converted to S2CellIds.
  Returns:
    A sequence of S2CellIds corresponding to the ids.
  '''
  return [s2.S2Cell(s2.S2CellId(cellid)) for cellid in list_ids]



def get_s2cover_for_s2polygon(s2polygon: s2.S2Polygon,
                level: int) -> Optional[Sequence]:
  '''Returns the cellids that cover the shape (point/polygon/polyline). 
  Arguments:
    s2polygon(S2Polygon): The S2Polygon to which S2Cells covering will be 
    performed.
  Returns:
    A sequence of S2Cells that completely cover the provided S2Polygon.
  '''

  if s2polygon is None:
    return None
  coverer = s2.S2RegionCoverer()
  coverer.set_min_level(level)
  coverer.set_max_level(level)
  coverer.set_max_cells(100)
  covering = coverer.GetCovering(s2polygon)
  for cell in covering:
    assert cell.level() == level

  return covering


def s2polygon_from_shapely_point(shapely_point: Point) -> s2.S2Polygon:
  '''Converts a Shapely Point to an S2Polygon.
  Arguments:
    point(Shapely Point): The Shapely Point to be converted to
    S2Polygon.
  Returns:
    The S2Polygon equivelent to the input Shapely Point.
  '''

  y, x = shapely_point.y, shapely_point.x
  latlng = s2.S2LatLng.FromDegrees(y, x)
  return s2.S2Polygon(s2.S2Cell(s2.S2CellId(latlng)))



def s2cellid_from_point(point: Point) -> int:
  '''Converts point to the S2CellId.
  Arguments:
    point: The point to be converted to S2CellId.
  Returns:
    The S2CellId equivelent to the input point.
  '''
  y, x = point.y, point.x
  latlng = s2.S2LatLng.FromDegrees(y, x)
  return s2.S2CellId(latlng).id()


def s2point_from_coord_xy(coord: CoordsXY) -> s2.S2Point:
  '''Converts coordinates (longtitude and latitude) to the S2Point.
  Arguments:
    coord: The coordinates given as longtitude and
    latitude to be converted to S2Point.
  Returns:
    The S2Point equivelent to the input coordinates .
  '''

  # Convert coordinates (lon,lat) to s2LatLng.
  latlng = s2.S2LatLng.FromDegrees(coord[1], coord[0])

  return latlng.ToPoint()  # S2Point


def s2polygon_from_shapely_polygon(shapely_polygon: Polygon) -> s2.S2Polygon:
  '''Convert a Shapely Polygon to S2Polygon. 
  Arguments:
    shapely_polygon(Polygon): The Shapely Polygon to be
    converted to S2Polygon.
  Returns:
    The S2Polygon equivelent to the input Shapely Polygon.
  '''

  # Filter where shape has no exterior attributes (e.g. lines).
  if not hasattr(shapely_polygon.buffer(0.00005), 'exterior'):
    return
  else:
    # Add a small buffer for cases where cover doesn't work.
    list_coords = list(shapely_polygon.buffer(0.00005).exterior.coords)

  # Get list of points.
  s2point_list = list(map(s2point_from_coord_xy, list_coords))
  s2point_list = s2point_list[::-1]  # Counterclockwise.
  return s2.S2Polygon(s2.S2Loop(s2point_list))


def s2polygon_from_shapely_polyline(shapely_polyine: Polygon) -> s2.S2Polygon:
  '''Convert a shapely polyline to s2polygon. 
  Arguments:
    shapely_polyine(Polygon): The Shapely Polygon which
    is a line to be converted to S2Polygon.
  Returns:
    The S2Polygon equivelent to the input Shapely Polygon.
  '''

  list_coords = list(shapely_polyine.exterior.coords)

  list_ll = []
  for lat, lng in list_coords:
    list_ll.append(s2.S2LatLng.FromDegrees(lat, lng))

  line = s2.S2Polyline()
  line.InitFromS2LatLngs(list_ll)

  return line

def cut(line: LineString, distance: float) -> Sequence[LineString]:
  '''Cut line in two at a distance from its 
  starting point. 
  Arguments:
    line: The to to be cut.
    distance: distance between line and its starting point.
  Returns:
    A list of lines.
  '''
  if distance <= 0.0 or distance >= line.length:
      return [LineString(line)] 
  coords = list(line.coords)
  for i, p in enumerate(coords):
      pd = line.project(Point(p))
      if pd == distance:
          return [
              LineString(coords[:i+1]),
              LineString(coords[i:])]
      if pd > distance:
          cp = line.interpolate(distance)
          return [
              LineString(coords[:i] + [(cp.x, cp.y)]),
              LineString([(cp.x, cp.y)] + coords[i:])]
  
  if coords[0]==coords[-1]: # It is a loop.
    cp = line.interpolate(distance)
    return [
            LineString(coords[:-1] + [(cp.x, cp.y)]),
            LineString([(cp.x, cp.y)] + coords[-1:])]

  return [LineString(line)] 

def plot_cells(cells: s2.S2Cell, location: Sequence[Point], zoom_level: int):
  '''Plot the S2Cell covering.'''

  # Create a map.
  map_osm = folium.Map(
    location=location, zoom_start=zoom_level, tiles='Stamen Toner')

  for cellid in cells:
    cellid = cellid[0]
    cell = s2.S2Cell(cellid)
    vertices = []
    for i in range(0, 4):
      vertex = cell.GetVertex(i)

      latlng = s2.S2LatLng(vertex)
      vertices.append((latlng.lat().degrees(),
               latlng.lng().degrees()))
    gj = folium.GeoJson(
      {
        "type": "Polygon",
        "coordinates": [vertices]
      },
      style_function={'weight': 1, 'fillColor': '#eea500'})
    gj.add_children(folium.Popup(cellid.ToToken()))
    gj.add_to(map_osm)

  filepath = 'visualization.html'
  map_osm.save(filepath)
  webbrowser.open(filepath, new=2)


def s2cellids_from_point(point: Point, level: int) -> Sequence[s2.S2CellId]:
  '''Get s2cell covering from shapely point (OpenStreetMaps Nodes). 
  Arguments:
    point(Point): a Shapely Point to which S2Cells.
    covering will be performed.
  Returns:
    A sequence of S2Cells that cover the provided Shapely Point.
  '''

  s2polygon = s2polygon_from_shapely_point(point)
  cellid = get_s2cover_for_s2polygon(s2polygon, level)[0]
  return [cellid]


def cellid_from_point(point: Point, level: int) -> int:
  '''Get s2cell covering from shapely point (OpenStreetMaps Nodes). 
  Arguments:
    point(Point): a Shapely Point to which S2Cells.
    covering will be performed.
  Returns:
    An id of S2Cellsid that cover the provided Shapely Point.
  '''

  s2polygon = s2polygon_from_shapely_point(point)
  cellids = get_s2cover_for_s2polygon(s2polygon, level)
  if cellids is None:
    sys.exit("S2cellid covering failed because the point is a None.")
  cellid = cellids[0]
  return cellid.id()


def s2cellids_from_polygon(polygon: Polygon, level: int) -> Optional[Sequence]:
  '''Get s2cell covering from shapely polygon (OpenStreetMaps Ways). 
  Arguments:
    polygon(Polygon): a Shapely Polygon to which S2Cells.
    covering will be performed..
  Returns:
    A sequence of S2Cells that cover the provided Shapely Polygon.
  '''

  s2polygon = s2polygon_from_shapely_polygon(polygon)
  return get_s2cover_for_s2polygon(s2polygon, level)


def cellids_from_polygon(polygon: Polygon, level: int) -> Optional[Sequence]:
  '''Get s2cell covering from shapely polygon (OpenStreetMaps Ways). 
  Arguments:
    polygon(Polygon): a Shapely Polygon to which S2Cells.
    covering will be performed..
  Returns:
    A sequence of S2Cells ids that cover the provided Shapely Polygon.
  '''

  s2polygon = s2polygon_from_shapely_polygon(polygon)
  s2cells = get_s2cover_for_s2polygon(s2polygon, level)
  return [cell.id() for cell in s2cells]


def cellid_from_polyline(polyline: Polygon, level: int) -> Optional[Sequence]:
  '''Get s2cell covering from shapely polygon that are lines (OpenStreetMaps
  Ways of streets).  
  Arguments: 
    polyline(Polygon): The Shapely Polygon of a line to which S2Cells
    covering will be performed. 
  Returns: 
    A sequence of s2Cells that cover the provided Shapely Polygon.
  '''

  s2polygon = s2polygon_from_shapely_polyline(polyline)
  return get_s2cover_for_s2polygon(s2polygon, level)


def get_bearing(start: Point, goal: Point) -> float:
  """Get the bearing (heading) from the start lat-lon to the goal lat-lon.
  Arguments:
    start: The starting point.
    goal: The goal point.
  Returns:
    The geospatial bearing when heading from the start to the goal. The 
    bearing angle given by azi1 (azimuth) is clockwise relative to north, so 
    a bearing of 90 degrees is due east, 180 is south, and 270 is west.
  """
  solution = geographiclib.geodesic.Geodesic.WGS84.Inverse(
    start.y, start.x, goal.y, goal.x)
  return solution['azi1'] % 360


def get_distance_km(start: Point, goal: Point) -> float:
  """Returns the geodesic distance (in kilometers) between start and goal.
  This distance is direct (as the bird flies), rather than based on a route
  going over roads and around buildings.
  """
  return geodesic(start.coords, goal.coords).km

def concat_numbers(n_1: int, n_2: int) -> int:
  '''Return the concatenation of two numbers.
  Arguments:
    n_1: The first number to be concatenated.
    n_2: The second number to be concatenated.
  Returns:
    A concatenated int of the two numbers
  '''
  return int(str(n_1) + str(n_2))


def get_distance_m(start: Point, goal: Point) -> float:
  """Returns the geodesic distance (in meters) between start and goal.
  This distance is direct (as the bird flies), rather than based on a route
  going over roads and around buildings.
  """
  return geodesic(start.coords, goal.coords).m

def tuple_from_point(point: Point) -> CoordsYX:
  '''Convert a Point into a tuple, with latitude as first element, and 
  longitude as second.
  Arguments:
    point(Point): A lat-lng point.
  Returns:
    A lat-lng coordinates.
  '''

  return CoordsYX(point.y, point.x)


def list_xy_from_point(point: Point) -> Sequence[float]:
  '''Convert a Point into a sequence, with longitude as first element, and 
  latitude as second.
  Arguments:
    point(Point): A lat-lng point.
  Returns:
    A lng-lat Sequence[float, float].
  '''

  return [point.x, point.y]


def list_yx_from_point(point: Point) -> Sequence[float]:
  '''Convert a Point into a sequence, with latitude as first element, and 
  longitude as second.
  Arguments:
    point(Point): A lat-lng point.
  Returns:
    A lat-lng Sequence[float, float].
  '''

  return [point.y, point.x]


def midpoint(p1: Point, p2: Point) -> Point:
  '''Get the midpoint between two points.
  Arguments:
    p1(Point): A lat-lng point.
    p2(Point): A lat-lng point.
  Returns:
    A lat-lng Point.
  '''
  return Point((p1.x+p2.x)/2, (p1.y+p2.y)/2)


def check_if_geometry_in_polygon(geometry: Any, poly: Polygon) -> Polygon:
  '''Check if geometry is intersects with polygon.
  Arguments:
    geometry: The geometry to check intersection against a polygon.
    poly: The polygon that to check intersection against a geometry.
  Returns:
    A lat-lng Point.
  '''
  if isinstance(geometry, Point):
    return poly.contains(geometry)
  else:
    geometry['geometry'].intersects(poly)

def get_distance_between_geometries(geometry: Any, point: Point) -> float:
  '''Calculate the distance between point and polygon in meters.
  Arguments:
    route: The line that length calculation will be performed on.
    point: point to measure distance from polygon.
  Returns:
    The distance between point and polygon in meters.
  '''
  if isinstance(geometry, Point):
    return get_distance_between_points(geometry, point)
  else:
    return get_distance_between_point_to_geometry(geometry, point)

def get_distance_between_point_to_geometry(
  geometry: Any, point: Point) -> float:
  '''Calculate the distance between point and polygon in meters.
  Arguments:
    route: The line that length calculation will be performed on.
    point: point to measure distance from polygon.
  Returns:
    The distance between point and polygon in meters.
  '''
  dist_min = float("Inf")
  if isinstance(geometry, MultiPolygon):
    coords = [coord for poly in geometry for coord in poly.exterior.coords]
  elif isinstance(geometry, Polygon):
    coords = geometry.exterior.coords
  else:
    coords = geometry.coords
  for coord in coords:
    point_current = Point(coord)
    dist = get_distance_between_points(point, point_current)
    if dist_min > dist:
      dist_min = dist
  return dist_min


def get_line_length(line: LineString) -> float:
  '''Calculate the length of a line in meters.
  Arguments:
    line: The line to calculate its length.
  Returns:
    The length of the line in meters
  '''
  dist = 0
  point_1 = Point(line.coords[0])
  for coord in line.coords[1:]:
    point_2 = Point(coord)
    dist += get_distance_between_points(point_1, point_2)
    point_1 = point_2

  return dist


def point_from_list_coord(coord: Sequence) -> Point:
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


def point_from_str_coord(coord_str: Text) -> Point:
  '''Converts coordinates in string format (latitude and longtitude) to Point. 
  E.g, of string '(40.715865, -74.037258)'.
  Arguments:
    coord: A lat-lng coordinate to be converted to a point.
  Returns:
    A point.
  '''
  list_coords_str = coord_str.replace("(", "").replace(")", "").split(',')
  coord = list(map(float, list_coords_str))

  return Point(coord[1], coord[0])


def coords_from_str_coord(coord_str: Text) -> CoordsYX:
  '''Converts coordinates in string format (latitude and longtitude) to 
  coordinates in a tuple format. 
  E.g, of string '(40.715865, -74.037258)'.
  Arguments:
    coord: A lat-lng coordinate to be converted to a tuple.
  Returns:
    A tuple of lat-lng.
  '''
  list_coords_str = coord_str.replace("(", "").replace(")", "").split(',')
  coord = list(map(float, list_coords_str))

  return CoordsYX(coord[0], coord[1])

def get_centers_from_s2cellids(
    s2cell_ids: Sequence[int64]) -> Sequence[Point]:
  """Returns the center latitude and longitude of s2 cells.
  Arguments:
    s2cell_ids: array of valid s2 cell ids. 1D array
  Returns:
    a list of shapely points of shape the size of the cellids list.
  """
  prediction_coords = []
  for s2cellid in s2cell_ids:
    s2_latlng = s2.S2CellId(int(s2cellid)).ToLatLng()
    lat = s2_latlng.lat().degrees()
    lng = s2_latlng.lng().degrees()
    prediction_coords.append(Point(lng, lat))
  return prediction_coords

def get_center_from_s2cellids(
    s2cell_ids: Sequence[int64]) -> Sequence[CoordsYX]:
  """Returns the center latitude and longitude of s2 cells.
  Arguments:
    s2cell_ids: array of valid s2 cell ids. 1D array
  Returns:
    a list of shapely points of shape the size of the cellids list.
  """
  prediction_coords = []
  for s2cellid in s2cell_ids:
    s2_latlng = s2.S2CellId(int(s2cellid)).ToLatLng()
    lat = s2_latlng.lat().degrees()
    lng = s2_latlng.lng().degrees()
    prediction_coords.append([lat, lng])
  return np.array(prediction_coords)
  
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

def get_distance_between_points(point_1: Point, point_2: Point) -> int:
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


def point_str_to_shapely_point(point_str: Text) -> Point:
  '''Converts point string to shapely point. 
  Arguments:
    point_str: The point string to be converted to shapely point. E.g, of 
    string 'Point(-74.037258 40.715865)'.
  Returns:
    A Point.
  '''
  point_str=point_str.split('(')[-1]
  point_str=point_str.split(')')[0]
  coords = point_str.split(" ")
  x, y = float(coords[0]), float(coords[1])
  return Point(x,y)

