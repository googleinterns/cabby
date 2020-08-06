from s2geometry import pywraps2 as s2
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
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import matplotlib
import folium
import webbrowser


class Map:

    def __init__(self, map_name):
        assert map_name == "Manhattan" or map_name == "Pittsburgh"
        self.graph = None

        if map_name == "Manhattan":
            self.place_polygon = shapely.wkt.loads(
                'POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'
            )

        else:
            self.place_polygon = shapely.geometry.box(
                miny=40.425, minx=-80.035, maxy=40.460, maxx=-79.930, ccw=True)

        self.poi, self.streets = self.__get_poi()
        self.__create_graph()

    def __get_poi(self):
        osm_items = []
        tags = {'name': True}

        osm_poi = ox.pois.pois_from_polygon(self.place_polygon, tags=tags)
        osm_poi = osm_poi[osm_poi['name'].notnull()]
        osm_poi_no_streets = osm_poi[osm_poi['highway'].isnull()]
        osm_poi_streets = osm_poi[osm_poi['highway'].notnull()]

        return osm_poi_no_streets, osm_poi_streets

    def cellid_from_s2shape(self, s2shape, level):
        # OpenStreetMaps Ways type to cellids

        coverer = s2.S2RegionCoverer()
        coverer.set_min_level(level)
        coverer.set_max_level(level)
        coverer.set_max_cells(100)
        covering = coverer.GetCovering(s2shape)
        for cell in covering:
            assert cell.level() == level

        return covering

    def latlng_from_point(self, point):
        '''Returns the a lat-lng 
        Arguments:
        point(Point): A lat-lng point.
        Returns:
        The lat-lng 
        '''

        return point.y, point.x

    def s2point_from_shapely_point(self, shapely_point):
        y, x = shapely_point.y, shapely_point.x
        latlng = s2.S2LatLng.FromDegrees(y, x)
        return latlng.ToPoint()

    def s2polygon_from_shapely_point(self, shapely_point):
        y, x = shapely_point.y, shapely_point.x
        latlng = s2.S2LatLng.FromDegrees(y, x)
        return s2.S2Polygon(s2.S2Cell(s2.S2CellId(latlng)))

    def cellid_from_point(self, s2_point, level):
        # OpenStreetMaps Nodes type to cellids
        return s2.S2CellId(s2_point, level)

    def s2point_from_coord_xy(self, coord):
        latlng = s2.S2LatLng.FromDegrees(coord[1], coord[0])
        return latlng.ToPoint()

    def latlng_from_coord_xy(self, coord):
        latlng = s2.S2LatLng.FromDegrees(coord[1], coord[0])
        return latlng

    def s2polygon_from_shapely_polygon(self, shapely_polygon):
        if not hasattr(shapely_polygon.buffer(0.00005), 'exterior'):
            return
        else:
            list_coords = list(shapely_polygon.buffer(
                0.00005).exterior.coords)

        s2point_list = list(map(self.s2point_from_coord_xy, list_coords))
        s2point_list = s2point_list[::-1]  # Counterclockwise
        return s2.S2Polygon(s2.S2Loop(s2point_list))

    def s2polygon_from_shapely_polyline(self, shapely_polygon):

        list_coords = list(shapely_polygon.exterior.coords)

        list_ll = []
        for lat, lng in list_coords:
            list_ll.append(s2.S2LatLng.FromDegrees(lat, lng))

        line = s2.S2Polyline()
        line.InitFromS2LatLngs(list_ll)

        return line

    def cellid_from_geometry(self, geo, level):
        assert isinstance(geo, Point) or isinstance(geo, Polygon), type(geo)
        if isinstance(geo, Point):
            s2_polygon = self.s2polygon_from_shapely_point(geo)
        else:
            s2_polygon = self.s2polygon_from_shapely_polygon(geo)
            if s2_polygon is None:
                return None
        return self.cellid_from_s2shape(s2_polygon, level)

    def plot_cells(self, cells):
        # create a map
        map_osm = folium.Map(
            location=[40.7434, -73.9847], zoom_start=12, tiles='Stamen Toner')

        def style_function(x): return {'weight': 1, 'fillColor': '#eea500'}

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
            gj = folium.GeoJson({"type": "Polygon", "coordinates": [
                                vertices]}, style_function=style_function)
            gj.add_children(folium.Popup(cellid.ToToken()))
            gj.add_to(map_osm)

        filepath = 'visualization.html'
        map_osm.save(filepath)
        webbrowser.open(filepath, new=2)

    def cellid_from_geometry_streets(self, geo, level):
        assert isinstance(geo, Point) or isinstance(geo, Polygon), type(geo)
        if isinstance(geo, Point):
            s2_shape = self.s2polygon_from_shapely_point(geo)
        else:
            s2_shape = self.s2polygon_from_shapely_polyline(geo)
        return self.cellid_from_s2shape(s2_shape, level)

    def add_poi_to_graph(self, row):
        cells = row.cellids
        poi = row.osmid
        self.graph.add_poi(cells, poi)

    def add_street_to_graph(self, row):
        cells = row.cellids
        street = row.osmid
        self.graph.add_street(cells, street)

    def __create_graph(self):
        level = 18

        # Get cellids for POI.
        self.poi['cellids'] = self.poi['geometry'].apply(
            self.cellid_from_geometry, args=[level])
        self.poi = self.poi[self.poi['cellids'].notnull()]
        # self.plot_cells(self.poi['cellids'].tolist())

        # Get cellids for streets
        self.streets['cellids'] = self.streets['geometry'].apply(
            self.cellid_from_geometry_streets, args=[level])
        self.streets = self.streets[self.streets['cellids'].notnull()]

        # Create graph.
        self.graph = Graph()

        # Add POI to graph
        self.poi[['cellids', 'osmid']].apply(self.add_poi_to_graph, axis=1)

        # Add street to graph
        self.streets[['cellids', 'osmid']].apply(
            self.add_street_to_graph, axis=1)

        '''
        TODO: 
        1. add streets to graph -done
        2. add multiple cell levels
        '''


if __name__ == "__main__":

    pittsburgh_map = Map("Pittsburgh")
    print("END")
