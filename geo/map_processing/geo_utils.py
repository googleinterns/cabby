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

from s2geometry import pywraps2 as s2
from s2geometry.pywraps2 import S2Point, S2Polygon, S2Polyline, S2Cell
from typing import Tuple, Sequence
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
import folium
import webbrowser
from typing import Optional


def get_s2cover_for_s2polygon(s2polygon: S2Polygon,
                              level: int) -> Optional[Sequence]:
    '''Returns the cellids that cover the shape (point\polygon\polyline). 
    Arguments:
    s2polygon(S2Polygon): an s2polygon.
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


def s2polygon_from_shapely_point(shapely_point: Point) -> S2Polygon:
    '''Converts a shapely Point to an s2Polygon.
    Arguments:
        point(Shapely Point): a Shapely type Point.
    Returns:
        The S2Polygon.
    '''

    y, x = shapely_point.y, shapely_point.x
    latlng = s2.S2LatLng.FromDegrees(y, x)
    return s2.S2Polygon(s2.S2Cell(s2.S2CellId(latlng)))


def s2point_from_coord_xy(coord: Tuple) -> S2Point:
    '''Converts coordinates (longtitude and latitude) to the s2point.
    Arguments:
        coord(S2Polygon): longtitude and latitude.
    Returns:
        An s2Point.
    '''

    # Convert coordinates (lon,lat) to s2LatLng.
    latlng = s2.S2LatLng.FromDegrees(coord[1], coord[0])

    return latlng.ToPoint()  # S2Point


def s2polygon_from_shapely_polygon(shapely_polygon: Polygon) -> S2Polygon:
    '''Convert a shapely polygon to s2polygon. 
    Arguments:
        shapely_polygon(Polygon): a shapely polygon.
    Returns:
        An s2Polygon.
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


def s2polygon_from_shapely_polyline(shapely_polyine: Polygon) -> S2Polygon:
    '''Convert a shapely polyline to s2polygon. 
    Arguments:
        shapely_polyine(Polygon): a shapely polygon.
    Returns:
        An s2Polygon.
    '''

    list_coords = list(shapely_polyine.exterior.coords)

    list_ll = []
    for lat, lng in list_coords:
        list_ll.append(s2.S2LatLng.FromDegrees(lat, lng))

    line = s2.S2Polyline()
    line.InitFromS2LatLngs(list_ll)

    return line


def plot_cells(cells: S2Cell):
    '''Plot the S2Cell covering.'''

    # create a map.
    map_osm = folium.Map(
        location=[40.7434, -73.9847], zoom_start=12, tiles='Stamen Toner')

    def style_function(x):
        return {'weight': 1, 'fillColor': '#eea500'}

    geoms = []
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
            style_function=style_function)
        gj.add_children(folium.Popup(cellid.ToToken()))
        gj.add_to(map_osm)

    filepath = 'visualization.html'
    map_osm.save(filepath)
    webbrowser.open(filepath, new=2)


def cellid_from_point(point: Point, level: int) -> Sequence:
    '''Get s2cell covering from shapely point (OpenStreetMaps Nodes). 
    Arguments:
        point(Point): a shapely point.
    Returns:
        A sequence of s2Cells.
    '''

    s2polygon = s2polygon_from_shapely_point(point)
    cellid = get_s2cover_for_s2polygon(s2polygon, level)[0]
    return [cellid]


def cellid_from_polygon(polygon: Polygon, level: int) -> Optional[Sequence]:
    '''Get s2cell covering from shapely polygon (OpenStreetMaps Ways). 
    Arguments:
        polygon(Polygon): a shapely Polygon.
    Returns:
        A sequence of s2Cells.
    '''

    s2polygon = s2polygon_from_shapely_polygon(polygon)
    return get_s2cover_for_s2polygon(s2polygon, level)


def cellid_from_polyline(polyline: Polygon, level: int) -> Optional[Sequence]:
    '''Get s2cell covering from shapely polygon that are lines (OpenStreetMaps
    Ways of streets).  
    Arguments: polyline(Polygon): a shapely Polygon of a street. Returns: A
        sequence of s2Cells.
    '''
    
    s2polygon = s2polygon_from_shapely_polyline(polyline)
    return get_s2cover_for_s2polygon(s2polygon, level)
