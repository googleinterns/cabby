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

import networkx as nx
import pandas as pd

from typing import Dict, Tuple, Sequence, Text, Any

from cabby.geo.map_processing.map_structure import Map
import cabby.geo.util as util
from cabby.data.wikidata import query
from cabby.geo.regions import Region, get_region

def convert_pandas_df_to_metagraph(
  df:Any, source_column: Text, target_column: Text,
  source_metadata_columns: Dict[Text, Text],
  target_metadata_columns: Dict[Text, Text],
  edge_attribute_columns: Sequence[Text]) -> Any:
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

def get_osmid_from_wd_relations_row(row: pd.Series):
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


def add_conceptual_nodes_and_edges(graph: nx.Graph,
                                   poi_map: Dict[Text, int],
                                   wd_relations: pd.DataFrame):
  for _, row in wd_relations.iterrows():
    place_node_id = poi_map[row["place"]]
    graph.add_edge(row["instanceLabel"], place_node_id)
    graph.add_edge(place_node_id, row["instanceLabel"])

def construct_human_readable_name(poi_row: pd.Series) -> Text:
    name = ""
    if isinstance(poi_row["name"], str):
        name = poi_row["name"]
    elif isinstance(poi_row["wikipedia"], str):
        name = poi_row["wikipedia"][3:]
    elif isinstance(poi_row["brand"], str):
        name = poi_row["brand"]
    elif isinstance(poi_row["shop"], str):
        name = poi_row["shop"]
    elif isinstance(poi_row["tourism"], str):
        name = poi_row["tourism"]
    elif isinstance(poi_row["amenity"], str):
        name = poi_row["amenity"]
    return "%s_%s" % (name, poi_row["osmid"])

def convert_multidi_to_weighted_undir_graph(
  in_graph: nx.MultiDiGraph) -> nx.Graph:
  out_graph = nx.Graph()
  for u, v in in_graph.edges():
      if out_graph.has_edge(u, v):
          out_graph[u][v]['weight'] += 1.0
      else:
          out_graph.add_edge(u, v, weight=1.0)
  return out_graph

def construct_metagraph(region: Region,
                        s2_level: int,
                        s2_node_levels: Sequence[int],
                        base_osm_map_filepath: Text) -> nx.Graph:
  # Get relation data and add to existing graph.
  wd_relations = query.get_geofenced_wikidata_relations(region,
                                                        extract_qids=True)
  osm_map = Map(region, s2_level, base_osm_map_filepath)
  update_osm_map(osm_map, wd_relations)

  # Construct initial metagraph and human-readable node names.
  metagraph = convert_multidi_to_weighted_undir_graph(osm_map.nx_graph)
  osmid_to_name = {}
  name_to_point = {}
  name_to_wikidata = {}
  for _, row in osm_map.poi.iterrows():
    name = construct_human_readable_name(row)
    assert isinstance(name, str)
    osmid_to_name = {row["osmid"]: name}
    name_to_point = {name: row["geometry"]}
    if isinstance(row["wikidata"], str):
      name_to_wikidata[name] = row["wikidata"]
  nx.relabel.relabel_nodes(metagraph, osmid_to_name,
                           copy=False)

  # Set useful node attributes.
  name_to_osmid = {osmid: name for name, osmid in osmid_to_name.items()}
  nx.set_node_attributes(metagraph, name_to_osmid, name="osmid")
  nx.set_node_attributes(metagraph, name_to_point, name="geometry")
  nx.set_node_attributes(metagraph, name_to_wikidata, name="wikidata")

  # Add conceptual edges from wd_relations.
  wikidata_to_name = {wikidata: name for
                      name, wikidata in name_to_wikidata.items()}
  for _, row in wd_relations.iterrows():
    place_node_name = wikidata_to_name[row["place"]]
    concept_node_name = "%s_%s" % (row["instanceLabel"], row["instance"])
    metagraph.add_edge(place_node_name, concept_node_name, weight=1.0)

  # S2 cell nodes and edges.
  edges_to_add = []
  for node, data in metagraph.nodes.data():
    if not data or "geometry" not in data:
      continue
    geometry = data["geometry"]
    for level in s2_node_levels:
      s2_cell_id = util.cellid_from_point(geometry, level)
      edges_to_add.append((node, s2_cell_id))
  metagraph.add_edges_from(edges_to_add)
  return metagraph