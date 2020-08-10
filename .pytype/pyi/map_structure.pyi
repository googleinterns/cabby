# (generated with --quick)

import graph
from typing import Any, Optional, Sequence, Type

Graph: Type[graph.Graph]
Point: Any
Polygon: Any
box: Any
ox: module
wkt: module

class Map:
    graph: graph.Graph
    place_polygon: Any
    poi: Any
    streets: Any
    def __init__(self, map_name: str, level: int) -> None: ...
    def create_graph(self, level: int) -> None: ...
    def get_poi(self) -> Sequence: ...

def cellid_from_point(point, level: int) -> Sequence: ...
def cellid_from_polygon(polygon, level: int) -> Optional[Sequence]: ...
def cellid_from_polyline(polyline, level: int) -> Optional[Sequence]: ...
