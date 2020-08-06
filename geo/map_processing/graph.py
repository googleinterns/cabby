from s2geometry import pywraps2 as s2
import networkx as nx


NUM_FACES = 6
MAX_LEVEL = 30
POS_BITS = 2 * MAX_LEVEL + 1
START_BIT = POS_BITS - 2

'''
TODO 
1. add adjacent cells connections
2. change graph to networkx
'''


class Node:
    def __init__(self, parent):
        self.parent = parent
        self.neighbors = [None, None, None, None, None, None, None, None]
        self.children = [None, None, None, None]
        self.poi = []
        self.streets = []
        self.level = 0
        if self.parent:
            self.level = self.parent.level+1


class Graph:
    def __init__(self):
        self.faces = tuple([Node(None) for x in range(0, NUM_FACES)])

    def get_cell_neighbors(self, cell):
        curr_node = self.faces[cell.face()]
        last_bit = POS_BITS - 2 * cell.level()
        cellid = cell.id()
        adjacent = cell.next()

    def add_street(self, cells, street):
        if isinstance(cells, s2.S2CellId):
            cells = [cells]
        for cell in cells:
            curr_node = self.faces[cell.face()]
            last_bit = POS_BITS - 2 * cell.level()
            cellid = cell.id()

            for i in range(START_BIT, last_bit-1, -2):
                lvlVal = (cellid >> i) & 3  # bits shift

                if not curr_node.children[lvlVal]:
                    curr_node.children[lvlVal] = Node(curr_node)

                curr_node = curr_node.children[lvlVal]

            curr_node.streets.append(street)

    def add_poi(self, cells, poi):
        if isinstance(cells, s2.S2CellId):
            cells = [cells]
        for cell in cells:
            curr_node = self.faces[cell.face()]
            last_bit = POS_BITS - 2 * cell.level()
            cellid = cell.id()

            for i in range(START_BIT, last_bit-1, -2):
                lvlVal = (cellid >> i) & 3

                if not curr_node.children[lvlVal]:
                    curr_node.children[lvlVal] = Node(curr_node)

                curr_node = curr_node.children[lvlVal]

            curr_node.poi.append(poi)

    def search(self, cell):
        curr_node = self.faces[cell.face()]
        last_bit = POS_BITS - 2 * cell.level()
        accum_poi = []
        cellid = cell.id()

        for i in range(START_BIT, last_bit-1, -2):
            lvlVal = (cellid >> i) & 3

            curr_node = curr_node.children[lvlVal]
            if not curr_node:
                return accum_poi

            if curr_node.poi:
                accum_poi = accum_poi + curr_node.poi

        return accum_poi
