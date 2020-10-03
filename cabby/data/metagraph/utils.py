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

def convert_pandas_df_to_metagraph(df, source_column, target_column,
                                   source_metadata_columns, target_metadata_columns,
                                   edge_attribute_columns):
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