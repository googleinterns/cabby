# Rendezvous(RVS): Location Finding using Nearby Landmarks

## Overview
The task we are considering is one where we have descriptions of goal locations relative to a starting point, and 1 or more landmarks. We will produce these synthetically, based on landmarks available in Wikidata and Open Street Maps for both New York City and Pittsburgh (the regions covered by the StreetLearn dataset).  As an example, starting in Bryant Park in NYC and with the goal location being the CVS pharmacy on 5th and 32nd, provide a description such as “the pharmacy one block past the Empire State Building.” If the starting point were Madison Square Park, then the description could be “the pharmacy a block before you get to the Empire State Building.”

## Generating Samples
To create the RVS dataset 3 steps are taken: (1) create an OSM graph of composed of entities from the region; (2) generate spatial samples based on OSM graph; (3) generate instructions based on spatial sample. 

### (1) Creating an OSM-based graph
```
bazel-bin/cabby/geo/map_processing/map_processor --region REGION --min_s2_level LEVEL --directory DIRECTORY_TO_MAP
```
### (2) Generating spatial samples
```
bazel-bin/cabby/geo/sample_poi --region REGION --min_s2_level LEVEL --directory DIRECTORY_TO_MAP --path PATH_TO_SPATIAL_ITEMS.gpkg --n_samples NUMBER_OF_SAMPLES_TO_GENERATE
```

### (3) Generating RVS instructions
```
bazel-bin/cabby/rvs/generate_rvs --rvs_data_path PATH_TO_SPATIAL_ITEMS.gpkg --save_instruction_path PATH_TO_RVS.json 
```



