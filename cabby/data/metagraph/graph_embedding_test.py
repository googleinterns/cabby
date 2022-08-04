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

'''Tests for utils.py'''

import unittest

import os
from node2vec import Node2Vec
from gensim.models import KeyedVectors

from cabby.geo.map_processing import map_structure
from cabby.geo import regions
from cabby.data.metagraph import utils


class GeoSetTest(unittest.TestCase):

  def testEmbeddingGraph(self):

    map_name = 'RUN-map1'

    map = map_structure.Map(regions.get_region(map_name), 14)

    current_dir = os.getcwd()

    map.write_map(current_dir)


    graph = utils.construct_metagraph(region=regions.get_region(map_name),
                                      s2_level=14,
                                      s2_node_levels=[14, 15],
                                      base_osm_map_filepath=current_dir,
                                      )

    # Precompute probabilities and generate walks

    node2vec = Node2Vec(
      graph,
      dimensions=1,
      walk_length=2,
      num_walks=1,
      workers=1)

    # Embed nodes
    # Any keywords acceptable by gensim.Word2Vec can be passed,
    # `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
    model = node2vec.fit(window=1, min_count=1, batch_words=4)

    # Save embeddings for later use
    path_to_save_embedding = f'embedding_{map_name.lower()}.pth'
    model.wv.save_word2vec_format(path_to_save_embedding)

    cell_embedding = KeyedVectors.load_word2vec_format(path_to_save_embedding)

    self.assertIn('6814312340', cell_embedding)


if __name__ == "__main__":
  unittest.main()
