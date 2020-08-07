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
import networkx as nx
from typing import Dict, Tuple, Sequence

# S2 geometry constants.
NUM_FACES = 6 # The top level of the hierarchy includes six faces 
MAX_LEVEL = 30  
POS_BITS = 2 * MAX_LEVEL + 1 
START_BIT = POS_BITS - 2 


class MapNode:

    def __init__(self, parent):
        self.parent = parent
        self.neighbors = [None, None, None, None, None, None, None, None]
        self.children = [None, None, None, None]
        self.poi = []
        self.streets = []
        self.level = 0
        if self.parent:
            self.level = self.parent.level + 1


class Graph:

    def __init__(self):
        self.faces = tuple([MapNode(None) for x in range(0, NUM_FACES)])

    def get_cell_neighbors(self, cell: s2.S2Cell) -> Sequence:
        '''Get eight s2cell neighbors for a given cell. 
        Arguments:
            cell(S2Cell): an S2Cell.
        Returns:
            A sequence of eight S2Cell2.
        '''
        four_neighbors = cell.GetEdgeNeighbors()
        eight_neighbors = four_neighbors + [
            four_neighbors[0].next(), four_neighbors[3].next(),
            four_neighbors[1].prev(), four_neighbors[3].prev()
        ]
        return eight_neighbors

    def add_street(self, cells: Sequence, street: int):
        '''Add a street POI to multiple cells. 
        Arguments:
            cells(sequence): a sequence of s2cells.
            street(int): an osmid of the street.
        '''
        if isinstance(cells, s2.S2CellId):
            cells = [cells]
        for cell in cells:
            curr_node = self.faces[cell.face()]
            last_bit = POS_BITS - 2 * cell.level()
            cellid = cell.id()

            for i in range(START_BIT, last_bit - 1, -2):
                lvlVal = (cellid >> i) & 3

                if not curr_node.children[lvlVal]:
                    curr_node.children[lvlVal] = MapNode(curr_node)

                curr_node = curr_node.children[lvlVal]

            curr_node.streets.append(street)
            curr_node.neighbors = self.get_cell_neighbors(cell)

    def add_poi(self, cells: Sequence, poi: int):
        '''Add a POI to multiple cells. 
        Arguments:
            cells(sequence): a sequence of s2cells.
            poi(int): an osmid of the POI.
        '''
        if isinstance(cells, s2.S2CellId):
            cells = [cells]
        for cell in cells:
            curr_node = self.faces[cell.face()]
            last_bit = POS_BITS - 2 * cell.level()
            cellid = cell.id()

            for i in range(START_BIT, last_bit - 1, -2):
                lvlVal = (cellid >> i) & 3

                if not curr_node.children[lvlVal]:
                    curr_node.children[lvlVal] = MapNode(curr_node)

                curr_node = curr_node.children[lvlVal]

            curr_node.poi.append(poi)
            curr_node.neighbors = self.get_cell_neighbors(cell)

    def search(self, cell: s2.S2Cell) -> Sequence:
        '''Get all POI for a specific cell. 
        Arguments:
            cell(S2Cell): an s2cell.
        Returns:
            A sequence of POI.

        '''
        curr_node = self.faces[cell.face()]
        last_bit = POS_BITS - 2 * cell.level()
        accum_poi = []
        cellid = cell.id()

        for i in range(START_BIT, last_bit - 1, -2):
            lvlVal = (cellid >> i) & 3

            curr_node = curr_node.children[lvlVal]
            if not curr_node:
                return accum_poi

            if curr_node.poi:
                accum_poi = accum_poi + curr_node.poi

        return accum_poi
