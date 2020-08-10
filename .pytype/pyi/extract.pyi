# (generated with --quick)

from typing import Any, Union

Point: Any
add_lat: Union[float, int]
add_lon: Union[float, int]
all_ids: list
i: int
idx: int
item: Any
items: Any
items_all: list
j: int
max_lat: float
max_lon: float
min_lat: float
min_lon: float
place_polygon: Any
point: Any
requests: module
shapely: module
wikdata_ids: list
wikipedia_found: list
x: Any
y: Any

def get_wikidata_id_from_wikipedia_id(ID) -> list: ...
def get_wikipedia_by_geosearch(longitude, latitude, radius, limit = ...) -> Any: ...
