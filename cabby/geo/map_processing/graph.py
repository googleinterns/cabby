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


from typing import Dict, Optional, Tuple, Sequence

import networkx as nx
from s2geometry import pywraps2 as s2


# S2 geometry constants.
NUM_FACES = 6  # The top level of the hierarchy includes six faces.
MAX_LEVEL = 30

# There are 60 bits reserved for the position along the Hilbert curve and there
# is one termination bit. The calculation: each level requires two bits (4
# children) + one is the termination bit.
NUM_POS_BITS = 2 * MAX_LEVEL + 1

# The begining of the position bits minus the two unique cell bits of first
# level.
START_BIT = NUM_POS_BITS - 2


class MapNode:

    # TODO(jasonbaldridge): Init using builder pattern.
    def __init__(self, parent):
        self.parent = parent
        self.neighbors = [None, None, None, None, None, None, None, None]
        self.children = [None, None, None, None]
        self.poi = []
        self.streets = []
        self.level = 0
        if self.parent is not None:
            self.level = self.parent.level + 1


class MapGraph:

    def __init__(self):
        self.faces = tuple([MapNode(None) for x in range(0, NUM_FACES)])

    def get_cell_neighbors(self, cell: s2.S2Cell) -> Sequence[MapNode]:
        '''Get eight S2Cell neighbors for a given cell. 
        Arguments:
            cell(S2Cell): The relative S2Cell to which the S2Cell neighbors will refer to.
        Returns:
            A sequence of eight S2Cell2 neighbors.
        '''

        four_neighbors = cell.GetEdgeNeighbors()
        eight_neighbors = four_neighbors + [
            four_neighbors[0].next(), four_neighbors[3].next(),
            four_neighbors[1].prev(), four_neighbors[3].prev()
        ]

        eight_neighbors_nodes = [self.search(cell) for cell in eight_neighbors]
        return eight_neighbors_nodes

    def add_street(self, cells: Sequence[s2.S2Cell], osmid_of_street: int):
        '''Add a street POI to multiple cells. 
        Arguments:
            cells(sequence): The S2Cells to which the street is to be added.
            osmid_of_street(int): The osmid of the street that is to be added to
            the S2Cells.
        '''

        for cell in cells:
            curr_node = self.faces[cell.face()]

            # Calculate the last position bit for the current level.
            # For each level 2 unique bits (=4 children) are reserved.
            last_bit = NUM_POS_BITS - 2 * cell.level()

            for i in range(START_BIT, last_bit - 1, -2):
                # Get value of unique bits in the current level.
                value_unique_bits = (cell.id >> i) & 3

                # Check if the child node exists and if not add a child node.
                if not curr_node.children[value_unique_bits]:
                    curr_node.children[value_unique_bits] = MapNode(curr_node)

                    # Connect the cell to its neighbors.
                    curr_node.neighbors = self.get_cell_neighbors(cell)

                # Descend one level by changing th current node to be the child
                # node.
                curr_node = curr_node.children[value_unique_bits]

            # Connect the cell to the street.
            curr_node.streets.append(osmid_of_street)

    def add_poi(self, cells: Sequence[s2.S2Cell], poi: int):
        '''Add a POI to multiple cells. 
        Arguments:
            cells(sequence): The S2Cells to which the POI will be added.
            poi(int): an osmid of the POI that will be added to the S2Cells.
        '''

        for cell in cells:
            curr_node = self.faces[cell.face()]

            # Calculate the last position bit for the current level.
            # For each level 2 unique bits (=4 children) are reserved.
            last_bit = NUM_POS_BITS - 2 * cell.level()

            for i in range(START_BIT, last_bit - 1, -2):
                # Get value of unique bits in the current level.
                value_unique_bits = (cell.id() >> i) & 3

                # Check if the child node exists and if not add a child node.
                if not curr_node.children[value_unique_bits]:
                    curr_node.children[value_unique_bits] = MapNode(curr_node)

                    # Connect the cell to its neighbors.
                    curr_node.neighbors = self.get_cell_neighbors(cell)

                # Descend one level by changing th current node to be the child
                # node.
                curr_node = curr_node.children[value_unique_bits]

            # Connect the cell to the POI.
            curr_node.poi.append(poi)

    def search(self, cell: s2.S2Cell) -> Optional[MapNode]:
        '''Get all POI for a specific cell. 
        Arguments:
            cell(S2Cell): The S2Cell to search for in the graph.
        Returns:
            The Node in the graph that contains the S2Cell.
        '''

        curr_node = self.faces[cell.face()]

        # Calculate the last position bit for the current level. For each level
        # 2 unique bits (=4 children) are reserved.
        last_bit = NUM_POS_BITS - 2 * cell.level()

        for i in range(START_BIT, last_bit - 1, -2):
            value_unique_bits = (cell.id() >> i) & 3

            curr_node = curr_node.children[value_unique_bits]

            if not curr_node:
                return None

        return curr_node
