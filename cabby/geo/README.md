# Generating spatial samples

## Overview
To create spatial samples 3 steps should be taken: (1) pick a path (start and end points and a route); (2) pick landmarks; (3)  calculate spatial-features such as cardinal direction.

### (1) Picking a path
To construct a path we first need to pick the start and end point. We start by selecting the end point. We randomly select a POI from the OSM-based  graph constructed, that is a small POI and has a tag key of a name. We define a small POI if its geometry covers 4 or less S2Cells of level 18 (no more than 152 meters long). We then pick the start point from the POI in the OSM-based graph based on the following conditions: 
1. A small POI
2. Within a path distance of 200-2000 meters from the end point
3. Has a tag key of name. 
We then pick a route that is the shortest distance between start and end point.  

### (2) Picking landmarks
We pick three landmarks that will be referenced in the instruction: (1) A landmark along the route (“on-path landmark”); (2) A landmark that is close to the goal location (“near landmark”); (3) a landmark that is on the same street as the goal location but beyond the route, such that if the agent keeps on walking beyond the goal it will reach that landmark (“beyond landmark”). To pick all three landmarks we prioritize POI that are most prominent. We define a prominent POI with the following order of importance (from the most to the least prominent): has wikipedia or wikidata link, is a brand, is a tourism attraction, is an amenity, is a shop. We randomly select all POI from the most prominent level found. 

### (3) Calculating spatial-features
In order to be able to refer to spatial features that are not landmarks, e.g, the number of blocks the agent needs to pass, calculation on the graph are required. The spatial features calculated are: (1) cardinal-directions between start and end point; (2) number of blocks between the on-path landmark and the goal location; (3) number of intersections between the on-path landmark and the goal location.

## Code for generating spatial samples
```
bazel-bin/cabby/geo/sample_poi --region REGION --min_s2_level LEVEL --directory DIRECTORY_TO_MAP --path PATH_TO_SPATIAL_ITEMS.gpkg --n_samples NUMBER_OF_SAMPLES_TO_GENERATE
```