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
import shapely
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
from typing import Tuple
from pandas import Series
from typing import Dict, Tuple, Sequence
from geo_utils import cellid_from_polyline, cellid_from_point, cellid_from_polygon
from shapely.geometry import box


class Map:

    def __init__(self, map_name: str, level: int):
        assert map_name == "Manhattan" or map_name == "Pittsburgh"
        self.graph = None

        if map_name == "Manhattan":
            self.place_polygon = shapely.wkt.loads(
                'POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'
            )

        else:  # Pittsburgh.
            self.place_polygon = box(
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


    def create_graph(self, level: int):
        '''Helper funcion for creating graph.'''

        # Get cellids for POI.
        self.poi['cellids'] = self.poi['geometry'].apply(lambda x: cellid_from_point(
            x, level) if isinstance(x, Point) else cellid_from_polygon(x, level))

        # Get cellids for streets.
        self.streets['cellids'] = self.poi['geometry'].apply(lambda x: cellid_from_point(
            x, level) if isinstance(x, Point) else cellid_from_polyline(x, level))

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
            lambda x: self.graph.add_street(x.cellids, x.osmid),
            axis=1)
