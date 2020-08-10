# (generated with --quick)

import typing
from typing import Any, Sequence, Type

Counter: Type[typing.Counter]
GeoDataFrame: Any
Graph: Any
MultiDiGraph: Any
Point: Any
Polygon: Any
S2Cell: Any
S2Point: Any
S2Polygon: Any
S2Polyline: Any
Series: Any
box: Any
cellid_from_point: Any
cellid_from_polygon: Any
cellid_from_polyline: Any
matplotlib: module
np: module
nx: module
ox: module
pd: module
plt: module
s2: module
shapely: module
time: module

class Map:
    graph: Any
    place_polygon: Any
    poi: Any
    streets: Any
    def __init__(self, map_name: str, level: int) -> None: ...
    def create_graph(self, level: int) -> None: ...
    def get_poi(self) -> Sequence: ...
