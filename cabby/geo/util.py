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

import folium
from functools import partial
import geographiclib
from geopy.distance import geodesic
import numpy as np
import pyproj
from s2geometry import pywraps2 as s2
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import box, mapping, LineString
from shapely.ops import transform
from typing import Optional, Tuple, Sequence, Any, Text
import webbrowser

UTM_CRS = 32633  # UTM Zones (North).
WGS84_CRS = 4326  # WGS84 Latitude/Longitude.


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


def s2cellid_from_coord_yx(coords: Tuple[float, float]) -> s2.S2CellId:
  '''Converts coordinates (latitude and longtitude) to the S2CellId.
  Arguments:
    coord(Tuple): The coordinates given as latitude and
    longtitude to be converted to S2CellId.
  Returns:
    The S2CellId equivelent to the input coordinates.
  '''
  y, x = coords[0], coords[0]
  latlng = s2.S2LatLng.FromDegrees(y, x)
  return s2.S2CellId(latlng)


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


def s2point_from_coord_xy(coord: Tuple) -> s2.S2Point:
  '''Converts coordinates (longtitude and latitude) to the S2Point.
  Arguments:
    coord(Tuple): The coordinates given as longtitude and
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


def s2cellid_from_point(point: Point, level: int) -> Sequence[s2.S2CellId]:
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
  cellid = get_s2cover_for_s2polygon(s2polygon, level)[0]
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


def project_point_in_segment(line_segment: LineString, point: Point):
  """Projects point to line and check if the point projected is in a segment. 
  Args:
    line_segment: The line segment.
    point: The point to be projected on the line.
  Returns:
    1 if the projected point is in the segment and 0 if inot.
  """

  point = np.array(point.coords[0])

  line_point_1 = np.array(line_segment.coords[0])
  line_point_2 = np.array(line_segment.coords[len(line_segment.coords)-1])

  diff = line_point_2 - line_point_1
  diff_norm = diff/np.linalg.norm(diff, 2)

  projected_point = line_point_1 + diff_norm * \
    np.dot(point - line_point_1, diff_norm)

  projected_point = Point(projected_point)

  return 1 if line_segment.distance(projected_point) < 1e-8 else 0


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


def tuple_from_point(point: Point) -> Tuple[float, float]:
  '''Convert a Point into a tuple, with latitude as first element, and 
  longitude as second.
  Arguments:
    point(Point): A lat-lng point.
  Returns:
    A lat-lng Tuple[float, float].
  '''

  return (point.y, point.x)


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


def coords_from_str_coord(coord_str: Text) -> Tuple[float, float]:
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

  return (coord[0], coord[1])


def get_cell_center_from_s2cellids(
    s2cell_ids: Sequence[int]) -> Sequence[Tuple]:
  """Returns the center latitude and longitude of s2 cells.
  Args:
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

def project_point_in_segment(line_segment: LineString, point: Point):
  """Projects point to line and check if the point projected is in a segment. 
  Args:
    line_segment: The line segment.
    point: The point to be projected on the line.
  Returns:
    1 if the projected point is in the segment and 0 if inot.
  """

  point = np.array(point.coords[0])

  line_point_1 = np.array(line_segment.coords[0])
  line_point_2 = np.array(line_segment.coords[len(line_segment.coords)-1])

  diff = line_point_2 - line_point_1
  diff_norm = diff/np.linalg.norm(diff, 2)

  projected_point = line_point_1 + diff_norm * \
    np.dot(point - line_point_1, diff_norm)

  projected_point = Point(projected_point)

  return 1 if line_segment.distance(projected_point) < 1e-8 else 0


def get_linestring_distance(line: LineString) -> int:
  '''Calculate the line length in meters.
  Arguments:
    route: The line that length calculation will be performed on.
  Returns:
    Line length in meters.
  '''
  project = partial(
    pyproj.transform,
    pyproj.Proj(WGS84_CRS),
    pyproj.Proj(UTM_CRS))

  trans_line = transform(project, line)

  return round(trans_line.length)

def point_str_to_shapely_point(point_str: Text) -> Point:
  '''Converts point string to shapely point. 
  Arguments:
    point_str: The point string to be converted to shapely point. E.g, of string 'Point(-74.037258 40.715865)'.
  Returns:
    A Point.
  '''
  point_str=point_str.split('(')[-1]
  point_str=point_str.split(')')[0]
  coords = point_str.split(" ")
  x, y = float(coords[0]), float(coords[1])
  return Point(x,y)