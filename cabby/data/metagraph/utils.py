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
      osmid = hash(''.join(list(row.values)))
      wd_query = query.get_geofenced_wikidata_items_by_qid(row.place)
      new_df = pd.DataFrame(data={
          'name': row.placeLabel,
          'geometry': [util.point_str_to_shapely_point(wd_query[0]['point']['value'])],
          'osmid': [osmid]
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

def construct_metagraph(region: Text,
                        s2_level: int,
                        base_osm_map_filepath: Text):
  wd_relations = query.get_geofenced_wikidata_relations(region,
                                                        extract_qids=True)
  osm_map = Map(region, s2_level, base_osm_map_filepath)
  update_osm_map(osm_map, wd_relations)
  return osm_map, wd_relations