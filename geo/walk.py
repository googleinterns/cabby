import networkx as nx
import osmnx as ox
from geopandas import GeoSeries
from networkx import MultiDiGraph



'''Library to support geographical computations .'''

def compute_route(start_point: tuple, end_point: tuple) -> list:
  '''Returns the shortest path between a starting and end point 

  Arguments:
    start_point(tuple): a lat-lng point
    end_point(tuple): a lat-lng point
  Returns:
    A list of points which construct the geometry of the path
  
  '''

  G = ox.graph_from_place('Manhattan, New York City, New York, USA') #change to bounding box
  nodes, streets = ox.graph_to_gdfs(G)
  
  #get closest nodes to points 
  orig=ox.get_nearest_node(G,start_point)
  dest=ox.get_nearest_node(G,end_point)


  #get route
  route = nx.shortest_path(G, orig, dest, weight='length') #shortest path
  return nodes[nodes['osmid'].isin(route)]['geometry'].tolist()