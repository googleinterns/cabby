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
'''Library to support weighted and type-specific random walks.'''

import collections

import networkx as nx
import numpy as np
from typing import Any, Dict, Sequence

from cabby.data.metagraph import utils as metagraph_utils

PLACE_NODE_TYPES = [metagraph_utils.OSM_LOC,
                    metagraph_utils.OSM_POI,
                    metagraph_utils.WD_POI,
                    metagraph_utils.PROJECTED_POI]

ALL_NODE_TYPES = [metagraph_utils.WD_CONCEPT,
                  metagraph_utils.S2] + PLACE_NODE_TYPES

TYPE_ATTR = "type"
WEIGHT_ATTR = "weight"

# Effective float zero for normalizing probability vectors.
PROBABILITY_SUM_MIN_WEIGHT = 1e-12

class ListSampler:
  '''Class for weighted sampling from an arbitrary list of items.

  If items and weights are provided at initialization, the ListSampler
  finalizes (normalizes probabilities and locks item set). If items and weights
  are added incrementally, `finalize` must be called before a sample is drawn.

  `items`: iterable of items, to be passed to `a` from
    `np.random.Generator.choice`. Default None; if not provided, items and
    weights can be added incrementally.
  `weights`: iterable of weights, to be normalized and then passed to `p` from
    `np.random.Generator.choice`. Default None; if not provided, items and
    weights can be added incrementally.
  '''

  def finalize(self):
      assert len(self._items) > 0, "must have some items to finalize"
      weight_sum = np.sum(self._weights)
      if weight_sum > PROBABILITY_SUM_MIN_WEIGHT:
        self._weights = np.array([w / weight_sum for w in weights])
      else:
        self._weights = np.ones(shape=(len(items),)) / float(len(items))
      self._is_finalized = True

  def __init__(self,
               items: Sequence[Any] = None,
               weights: Sequence[float] = None)
      assert len(items) == len(weights), "items and weights are different sizes"
      self._items = list(items)
      self._weights = list(weights)
      self._is_finalized = False

  def add_item(self, item, weight):
      assert self._is_finalized, "cannot add to finalized ListSampler"
      self._items.append(item)
      self._weights.append(weight)

  def sample_item(self, rng: np.random.Generator):
      return rng.choice(a=self._items, p=self._weights)

class MetagraphRandomWalker:
  '''Stores transition probabilities and generates random walks for metagraph.

  (Details...)
  
  `nx_graph`: An undirected networkx graph on which to compute random walks.
    Must be undirected and single-edge. Can be unweighted, but unweighted edges
    will be given a default weight of 1 when computing transition probabilities.
    This class is meant for the output of metagraph_utils.construct_metagraph.

  `distance_transformer`: A function for transforming distance-weighted edges. A
    distance-weighted edge is an edge between any pair of nodes which both have
    a `type` attribute contained in `PLACE_NODE_TYPES` above. Default None,
    which indicates no transformation.

  `use_weights`: If False, will ignore existing weights on the graph. Next nodes
    in random walks will be chosen uniformly-at-random from the neighbor set.
    Default true.

  `type_distribution`: A dictionary mapping from a node type in ALL_NODE_TYPES
    to a float. If specified (default None), the dictionary keys must be
    exactly those in ALL_NODE_TYPES, throwing AssertionError otherwise. The
    values of the map will be normalized to create a probability distribution
    over types. A random walker will first choose a node type, and then a node
    from its outgoing-edge distribution to that node type. If this map is not
    specified, the walker will ignore type, and choose nodes proportional to
    their outgoing weight.

  '''

  def _compute_sampler_map(self):
      '''Computes map of ListSamplers for random walks.

      If type_distribution was None, this will compute a simple map from node id
      to a ListSampler over all nodes, regardless of type. If type_distribution
      was a dict, this will compute a map from node id to a node type map, where
      each node type map is a dict from a node type (str) to a ListSampler over
      nodes of that type.
      '''
      self._sampler_map = {}
      for source, sdata in self._nx_graph.nodes(data=True):
        # Get source data
        target_view = self._nx_graph[source]
        source_type = sdata[TYPE_ATTR]
        source_is_place = source_type in PLACE_NODE_TYPES
        # Prepare transition struct
        if self._use_types:
            samplers = collections.defaultdict(ListSampler())
        else:
            samplers = ListSampler()
        # Iterate through adjacency lists and add samplers to sampler map.
        for target, tdict in target_view.items():
          target_type = self._nx_graph.nodes[target][TYPE_ATTR]
            if WEIGHT_ATTR in tdict and self._use_weights:
              weight = tdict[WEIGHT_ATTR]
                if self._dist_trans is not None:
                  if source_is_place and target_type in PLACE_NODE_TYPES:
                    weight = self._dist_trans(weight)
            else:
              weight = 1.0
            if self._use_types:
                samplers[target_type].add_item(target, weight)
            else:
                samplers.add_item(target, weight)
        if self._use_types:
            for node_type in ALL_NODE_TYPES:
                samplers[node_type].finalize()
        else:
            samplers.finalize()
        self._transition_map[source] = copy.deepcopy(samplers)

  def __init__(self, nx_graph: nx.Graph,
               distance_transformer: Any = None,
               use_weights: bool = True,
               type_distribution: Dict[str, float] = None):
    assert not nx.is_directed(nx_graph)
    assert (type_distribution is None or
            all([t in ALL_NODE_TYPES for t in list(type_distribution.keys())]))
    self._nx_graph = nx_graph
    self._dist_trans = distance_transformer
    self._use_weights = use_weights
    self._use_types = type_distribution is not None
    if self._use_types:
        self._type_sampler = ListSampler(
            items=ALL_NODE_TYPES,
            weights=[type_distribution(t) for t in ALL_NODE_TYPES])