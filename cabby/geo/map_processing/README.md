# OSM-based Graph

For each region we construct an OSM-based graph. 

To construct the graph we first extract OSM points of interest (POI) and streets from the OSM API. OSM entities(streets and POI) include various tags with a key and a value pair (e.g, key=name and value=Empire State Building). The POI extracted must include one or more of the following tag keys: name, shop, brand, tourism, amenity, wikipedia, wikidata. E.g., of a POI that will be extracted: {'tourism': 'viewpoint', 'osmid': 3826035511, 'geometry': Point(-80.00960449999999,40.44239110000001, 'element_type': node)}.


Second, we then project the POI on the streets extracted. For each POI a node is added with a tag name 'poi'. In addition if the projected point splits the edge it is projected on then an additional node is added with tage name 'projected-poi'. An edge added between the two nodes (POI and the projected\end of edge) with tage name 'poi'. The POI osmid change when added to the graph, an aditional prefix is added - '999'. E.g, an osmid 3826035511 will now be osmid 9993826035511. Projected nodes have the same osmid (with 999) but with an initial number (1-4) so if a POI with osmid 3826035511 (converted to 9993826035511) has 2 points projected on 2 different streets, then their osmids will be 19993826035511 and 29993826035511.


| ![](images/projected_poi.png) | 
|:--:| 
| *An example of POI projection on edge.* |


 Thus, the edges are the street and the nodes are elements of the street (e.g, intersection), a projected POI, or POI. The POI are only connected to the streets and not among themself. A POI can be one of the following geometries: point, line, polygon. The POI is connected to the closest streets. If it is a point geometry it can be only connected to one street - the closest one. If it is a line or a polygon then different parts of the geometry can be close to different streets and therefore it can be connected to multiple streets (up to 4 streets).  

The graph we construct is a directional graph with multiple edges between two nodes. Multiple edges might occur in cases where there are two paths between nodes or in case of a loop. It is also a weighted graph where the weight is the actual distance between the nodes in meters. 
The edges also have tags, e.g., key=highway and value=residential.


| ![](images/parallel_paths.png) | 
|:--:| 
| *An example of a case where there are parallel paths between two nodes (U and V).* |



| ![](images/loop.png) | 
|:--:| 
| *An example of a case where there is a loop in the graph.*|



| ![](images/graph.png) | 
|:--:| 
| *An example of a graph.*|


## Files in Map
+ Map.poi will be saved and loaded as region_poi.pkl. 
  - It includes all the POI extracted. 
  - It is a Geopandas.DataFrame class where each row is a single POI.
  - It can be joined with Map.nx_graph (region_graph.gpickle file) on the osmid. Thus a POI node in Map.nx_graph with osmid 1 will be found in Map.poi in one of the following ways: 

```
poi = poi[poi['osmid']==1]
```
or

```
poi = poi.loc[1]
```

+ Map.streets will be saved and loaded as region_streets.pkl
  - It includes streets extracted. 
  - It is a Geopandas.DataFrame class where each row is a single street.

+ Map.nx_graph will be saved and loaded as region_graph.gpickle
  - It includes map graph.
  - It is a networkx.MultiDiGraph class with nodes and edges.

+ Map.nodes will be saved and loaded as region_nodes.gpickle
  - It includes the nodes from the map graph.
  - It is a Geopandas.DataFrame class of the nodes in Map.nx_graph.

+ Map.edges will be saved and loaded as region_edges.gpickle
  - It includes the edges from the map graph.
  - It is a Geopandas.DataFrame class of the edges in Map.nx_graph.


## Code for creating an OSM-based graph
```
bazel-bin/cabby/geo/map_processing/map_processor --region REGION --min_s2_level LEVEL --directory DIRECTORY_TO_MAP
```
