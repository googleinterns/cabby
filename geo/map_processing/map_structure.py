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

from s2geometry import pywraps2 as s2
from s2geometry.pywraps2 import S2Point, S2Polygon, S2Polyline, S2Cell
import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame
from networkx import MultiDiGraph
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import pandas as pd
import shapely.wkt
import time
from collections import Counter
from graph import Graph
import matplotlib.pyplot as plt
import matplotlib
import folium
from typing import Tuple
from pandas import Series
from typing import Dict, Tuple, Sequence


class Map:

    def __init__(self, map_name: str, level: int):
        assert map_name == "Manhattan" or map_name == "Pittsburgh"
        self.graph = None

        if map_name == "Manhattan":
            self.place_polygon = shapely.wkt.loads(
                'POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'
            )

        else:  # Pittsburgh.
            self.place_polygon = shapely.geometry.box(
                miny=40.425, minx=-80.035, maxy=40.460, maxx=-79.930, ccw=True)

        self.poi, self.streets = self.get_poi()
        self.create_graph(level)

    def get_poi(self) -> Sequence:
        '''Helper funcion  for extracting POI for the defined place.'''

        osm_items = []
        tags = {'name': True}

        osm_poi = ox.pois.pois_from_polygon(self.place_polygon, tags=tags)
        osm_poi = osm_poi[osm_poi['name'].notnull()]
        osm_poi_no_streets = osm_poi[osm_poi['highway'].isnull()]
        osm_poi_streets = osm_poi[osm_poi['highway'].notnull()]

        return osm_poi_no_streets, osm_poi_streets

    def get_s2cover_for_s2polygon(self, s2polygon: S2Polygon,
                                  level: int) -> Sequence:
        '''Returns the cellids that cover the shape (point\polygon\polyline). 
        Arguments:
        s2polygon(S2Polygon): an s2polygon.
        Returns:
        A sequence of S2Cell.

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

    def s2polygon_from_shapely_point(self, shapely_point: Point) -> S2Polygon:
        '''Converts a shapely Point to an s2Polygon.
        Arguments:
            point(Shapely Point): a Shapely type Point.
        Returns:
            The S2Polygon.
        '''

        y, x = shapely_point.y, shapely_point.x
        latlng = s2.S2LatLng.FromDegrees(y, x)
        return s2.S2Polygon(s2.S2Cell(s2.S2CellId(latlng)))

    def s2point_from_coord_xy(self, coord: Tuple) -> S2Point:
        '''Converts coordinates (longtitude and latitude) to the s2point.
        Arguments:
            coord(S2Polygon): longtitude and latitude.
        Returns:
            An s2Point.

        '''
        # Convert coordinates (lon,lat) to s2LatLng.
        latlng = s2.S2LatLng.FromDegrees(coord[1], coord[0])

        return latlng.ToPoint()  # S2Point

    def s2polygon_from_shapely_polygon(self,
                                       shapely_polygon: Polygon) -> S2Polygon:
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
        s2point_list = list(map(self.s2point_from_coord_xy, list_coords))
        s2point_list = s2point_list[::-1]  # Counterclockwise.
        return s2.S2Polygon(s2.S2Loop(s2point_list))

    def s2polygon_from_shapely_polyline(self,
                                        shapely_polyine: Polygon) -> S2Polygon:
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

    def plot_cells(self, cells: S2Cell):
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

    def cellid_from_point(self, point: Point, level: int) -> Sequence:
        '''Get s2cell covering from shapely point (OpenStreetMaps Nodes). 
        Arguments:
            point(Point): a shapely point.
        Returns:
            A sequence of s2Cells.

        '''
        s2polygon = self.s2polygon_from_shapely_point(point)
        cellid = self.get_s2cover_for_s2polygon(s2polygon, level)[0]
        return [cellid]

    def cellid_from_polygon(self, polygon: Polygon, level: int) -> Sequence:
        '''Get s2cell covering from shapely polygon (OpenStreetMaps Ways). 
        Arguments:
            polygon(Polygon): a shapely Polygon.
        Returns:
            A sequence of s2Cells.

        '''
        s2polygon = self.s2polygon_from_shapely_polygon(polygon)
        return self.get_s2cover_for_s2polygon(s2polygon, level)

    def cellid_from_polyline(self, polyline: Polygon, level: int) -> Sequence:
        '''Get s2cell covering from shapely polygon that are lines (OpenStreetMaps Ways of streets). 
        Arguments:
            polyline(Polygon): a shapely Polygon of a street.
        Returns:
            A sequence of s2Cells.
        '''
        s2polygon = self.s2polygon_from_shapely_polyline(polyline)
        return self.get_s2cover_for_s2polygon(s2polygon, level)

    def create_graph(self, level: int):
        '''Helper funcion for creating graph.'''

        # Get cellids for POI.
        self.poi['cellids'] = self.poi['geometry'].apply(lambda x: self.cellid_from_point(
            x, level) if isinstance(x, Point) else self.cellid_from_polygon(x, level))

        # Get cellids for streets.
        self.streets['cellids'] = self.poi['geometry'].apply(lambda x: self.cellid_from_point(
            x, level) if isinstance(x, Point) else self.cellid_from_polyline(x, level))

        # Filter out entities that we didn't mange to get cellids covering.
        self.poi = self.poi[self.poi['cellids'].notnull()]
        self.streets = self.streets[self.streets['cellids'].notnull()]

        # Create graph.
        self.graph = Graph()

        # Add POI to graph.
        self.poi[['cellids', 'osmid']].apply(
            lambda x: self.graph.add_poi(x.cellids, x.osmid), axis=1)

        # Add street to graph.
        self.streets[['cellids', 'osmid']].apply(
            lambda x: self.graph.add_street_to_graph(x.cellids, x.osmid),
            axis=1)
