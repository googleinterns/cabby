# (generated with --quick)

from typing import Any, List, Optional, Sequence, Tuple

MAX_LEVEL: int
NUM_FACES: int
NUM_POS_BITS: int
START_BITS: int
nx: module
s2: module

class Graph:
    faces: Tuple[MapNode, ...]
    def __init__(self) -> None: ...
    def add_poi(self, cells: Sequence, poi: int) -> None: ...
    def add_street(self, cells: Sequence, osmid_of_street: int) -> None: ...
    def get_cell_neighbors(self, cell) -> Sequence: ...
    def search(self, cell) -> Optional[MapNode]: ...

class MapNode:
    children: List[None]
    level: Any
    neighbors: List[None]
    parent: Any
    poi: List[nothing]
    streets: List[nothing]
    def __init__(self, parent) -> None: ...
