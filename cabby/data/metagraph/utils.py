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

'''Utils for metagraph construction'''

from absl import logging

import collections
import networkx as nx
import numpy as np
import pandas as pd

from shapely.geometry import point
from typing import Any, Dict, Sequence

from cabby.geo.map_processing import map_structure
from cabby.data.wikidata import query
from cabby.geo import regions
from cabby.geo import util


DEFAULT_POI_READABLE_NAME = "poi"
DEFAULT_EDGE_WEIGHT = 1.0
PROJECTED_POI_2ND_CHAR = "#"
IMPORTANT_POI_METADATA_FIELDS = [
  "name", "wikipedia", "wikidata", "brand", "shop", "tourism", "amenity"
]

# Node type names
TYPE_OSM_LOC = "OSM_LOC"
TYPE_OSM_POI = "OSM_POI"
TYPE_WD_POI = "WD_POI"
TYPE_WD_CONCEPT = "WD_CONCEPT"
TYPE_S2 = "S2"
TYPE_PROJECTED_POI = "PROJECTED_POI"

# Type declarations
Map = map_structure.Map
Point = point.Point
Region = regions.Region

def convert_pandas_df_to_metagraph(
  df:Any, source_column: str, target_column: str,
  source_metadata_columns: Dict[str, str],
  target_metadata_columns: Dict[str, str],
  edge_attribute_columns: Sequence[str]) -> Any:
  '''Convert a pandas dataframe to a networkx multigraph
  Arguments:
    df: pandas dataframe
    source_column: column in df to use as source node label
    target_column: column in df to use as target node label
    source_metadata_columns: for source nodes, dict from column name to
                             metadata field name
    target_metadata_columns: for target nodes, dict from column name to
                             metadata field name
    edge_attribute_columns: list of columns in df to use as edge attributes
  Returns:
    g: networkx Graph holding all graph data.
  '''
  g = nx.convert_matrix.from_pandas_edgelist(df,
                                             source=source_column,
                                             target=target_column,
                                             edge_attr=edge_attribute_columns,
                                             create_using=nx.classes.multidigraph.MultiDiGraph)
  for metadata_column, metadata_name in source_metadata_columns.items():
      attribute_dict = dict(zip(df[source_column], df[metadata_column]))
      nx.set_node_attributes(g, attribute_dict, metadata_name)
  for metadata_column, metadata_name in source_metadata_columns.items():
      attribute_dict = dict(zip(df[target_column], df[metadata_column]))
      nx.set_node_attributes(g, attribute_dict, metadata_name)
  return g

def get_osmid_from_wd_relations_row(row: pd.Series) -> int:
  """Gets a fake Open Street Map ID (OSMID) for an arbitrary POI.
  This is used for POIs that were found in a Wikidata bounding box
  for a region, but for one reason or another were not pulled from OSM.
  Arguments:
    row: an arbitrary sequence of data on the POI. Should uniquely ID
      the POI to get a unique fake OSMID.
  Returns:
    osmid: an integer fake OSMID.
  """
  osmid = hash(''.join(list(row.values)))
  if osmid < 0:
    osmid *= -1
  return osmid

def update_osm_map(osm_map: Map,
                   wd_relations: pd.DataFrame):
  """Adds new POIs found in wikidata_relations to osm_map.
  Arguments:
    osm_map: map class containing all info about a region.
    wd_relations: dataframe of wikidata items in the region.
  Returns: (nothing)
  """
  # Compute QIDs present in wikidata_relations but not in osm_map.
  osm_qids = set(
    [qid for qid in osm_map.poi['wikidata'] if isinstance(qid, str)])
  wd_qids = set(wd_relations.place)
  missing_qids = wd_qids - osm_qids
  print("In update_osm_map: Found %d missing qids." % len(missing_qids))
  if len(missing_qids) == 0:
    return

  # For each missing QID, add the place to the osm_map data.
  print("In update_osm_map: Adding new QIDs to osm_map data.")
  print("In update_osm_map: POI table has %d rows before" % osm_map.poi.shape[0])
  already_added = set()
  for _, row in wd_relations.iterrows():
      if row.place not in missing_qids or row.place in already_added:
          continue
      already_added.add(row.place)
      osmid = get_osmid_from_wd_relations_row(row)
      wd_query = query.get_place_location_points_from_qid(
        row.place, location_only=True)
      new_df = pd.DataFrame(data={
          'name': row["placeLabel"],
          'geometry': [util.point_str_to_shapely_point(wd_query[0]['point']['value'])],
          'centroid': [util.point_str_to_shapely_point(wd_query[0]['point']['value'])],
          'osmid': [osmid],
          'wikidata': row["place"]
      }, index=[osmid])
      new_df.index.rename('osmid', inplace=True)
      osm_map.poi = osm_map.poi.append(new_df)
  print("In update_osm_map: POI table has %d rows after" % osm_map.poi.shape[0])

  # Update osm_map.
  print("In update_osm_map: Adding new POIs to graph (could take a while).")
  print("In update_osm_map: %d nodes in graph before adding." % (
    osm_map.nx_graph.number_of_nodes()))
  osm_map.add_poi_to_graph()
  print("In update_osm_map: %d nodes in graph after adding." % (
    osm_map.nx_graph.number_of_nodes()))

def convert_multidi_to_weighted_undir_graph(
  in_graph: nx.MultiDiGraph, agg_function: Any) -> nx.Graph:
  """Convert a graph with multiple edges to a graph with no multiple edges.
  Arguments:
    in_graph: graph with (potentially) multiple directed edges per node pair.
    agg_function: a function that takes an iterable of floats and returns
      a number. Applied to weights on multiple edges to produce one weight.
  Returns:
    out_graph: graph with weighted undirected edges.
  """
  edges = collections.defaultdict(dict)
  for node, neighbor_dict in in_graph.adjacency():
    for neighbor, edges_dict in neighbor_dict.items():
      id_pair = tuple(sorted([node, neighbor]))
      if 'weight' not in edges[id_pair]:
        edges[id_pair]['weight'] = []
      for _, edge_data in edges_dict.items():
        # Some input edges have artificially high weight [1] in to prevent
        # an agent from visiting the center of a POI. So, the input is also
        # given a "true_length" attribute [2] so that the graph construction
        # can use the actual distance.
        # [1] See cabby.geo.map_processing.map_structure.ADD_POI_DISTANCE
        # [2] See cabby.geo.map_processing.edge
        if 'true_length' in edge_data:
          weight = edge_data['true_length']
        else:
          weight = edge_data['length']
        edges[id_pair]['weight'].append(weight)
  for id_pair in edges:
    edges[id_pair]['weight'] = agg_function(edges[id_pair]['weight'])
  out_graph = nx.Graph()
  out_graph.add_edges_from([id_pair + (data,) for id_pair, data in edges.items()])
  return out_graph

def construct_metagraph(region: Region,
                        s2_level: int,
                        s2_node_levels: Sequence[int],
                        base_osm_map_filepath: str,
                        agg_function=np.sum) -> nx.Graph:
  """Builds metagraph from existing OSM data and Wikidata/S2 nodes in a region.
  The region, s2_level, and base_osm_filepath arguments are passed to the
  constructor of map_structure.Map.
  # TODO(palowitch): add detail on weight scale & weights for different types.
  Arguments:
    Region: region to build the graph on.
    s2_level: S2 level of the map_structure.Map to be loaded.
    s2_node_levels: iterable of S2 cell levels (ints) to add to the graph.
    base_osm_map_filepath: location of the map_structure.Map to be loaded.
    agg_function: a function that takes an iterable of floats and returns
      a number. Applied to weights on the multi-edge input OSM graph to produce
      a single weight value.
  Returns:
    metagraph: an nx.Graph with undirected edges and weights as described above.
  """
  # Step 0: Load the OSM graph and add extra wikidata-found places to it.
  wd_relations = query.get_geofenced_wikidata_relations(
    region, extract_qids=True)
  osm_map = Map(region, s2_level, base_osm_map_filepath)
  update_osm_map(osm_map, wd_relations)

  # Step 1: Convert the nx.MultiDiGraph into a weighted nx.Graph.
  metagraph = convert_multidi_to_weighted_undir_graph(osm_map.nx_graph,
                                                      agg_function)

  # Step 2: Add all geometries to the graph.
  for _, row in osm_map.nodes.iterrows():
    metagraph.nodes[row["osmid"]]["geometry"] = row["geometry"]
    metagraph.nodes[row["osmid"]]["type"] = TYPE_OSM_LOC
  for _, row in osm_map.poi.iterrows():
    if "geometry" in metagraph.nodes[row["osmid"]]:
      continue
    assert isinstance(row["geometry"], Point)
    metagraph.nodes[row["osmid"]]["geometry"] = row["geometry"]
    metagraph.nodes[row["osmid"]]["type"] = TYPE_OSM_POI

  # Step 3: Add important POI attributes.
  attributes_to_add = collections.defaultdict(dict)
  wikidata_to_nodeid = {}  # Needed for wikidata concept node additions.
  for _, row in osm_map.poi.iterrows():
    node_id = row["osmid"]
    for field in IMPORTANT_POI_METADATA_FIELDS:
      if isinstance(row[field], str):
        attributes_to_add[node_id][field] = row[field]
        if field == "wikidata":
          wikidata_to_nodeid[row[field]] = node_id
  nx.set_node_attributes(metagraph, values=attributes_to_add)

  # Step 4: Add conceptual nodes, attributes, and edges to the graph.
  attributes_to_add = collections.defaultdict(dict)
  for _, row in wd_relations.iterrows():
    # Add edge.
    place_node_id = wikidata_to_nodeid[row["place"]]
    concept_node_id = row["instance"]
    # TODO(palowitch): make a smarter (possibly configurable/programmatic) default weight.
    metagraph.add_edge(node_id, concept_node_id, weight=1.0)
    # Add attributes.
    attributes_to_add[place_node_id]["name"] = row["placeLabel"]
    attributes_to_add[place_node_id]["type"] = TYPE_WD_POI
    attributes_to_add[concept_node_id]["name"] = row["instanceLabel"]
    attributes_to_add[concept_node_id]["wikidata"] = row["instance"]
    attributes_to_add[concept_node_id]["type"] = TYPE_WD_CONCEPT
  nx.set_node_attributes(metagraph, values=attributes_to_add)

  # Step 5: Add S2 nodes and edges
  edges_to_add = []
  for node, data in metagraph.nodes.data():
    if "geometry" not in data:
      continue
    geometry = data["geometry"]
    for level in s2_node_levels:
      s2_cell_node_id = util.cellid_from_point(geometry, level)
      edges_to_add.append((node, s2_cell_node_id, {"weight": 1.0}))
      attributes_to_add[s2_cell_node_id]["type"] = TYPE_S2
  metagraph.add_edges_from(edges_to_add)
  nx.set_node_attributes(metagraph, values=attributes_to_add)

  # Step 6: Add types to projected POIs.
  attributes_to_add = collections.defaultdict(dict)
  for node in metagraph:
    nodedata = metagraph.nodes[node]
    node = str(node)
    if node[1] == PROJECTED_POI_2ND_CHAR:
      attributes_to_add[node]["type"] = TYPE_PROJECTED_POI
    else:
      assert "type" in nodedata, "node %s has no type and is not a projected POI" % node
  nx.set_node_attributes(metagraph, values=attributes_to_add)

  logging.info("Finished constructing graph")
  return metagraph