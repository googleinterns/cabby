# Check the types in everything.
pytype cabby

# Geo
bazel-bin/cabby/geo/geo_computation --orig_lat 40.749102 --orig_lon -73.984076 -dest_lat 40.748432 --dest_lon -73.982473
bazel-bin/cabby/geo/map_processing/map_processor --region DC --min_s2_level 18 --directory "./cabby/geo/map_processing/poiTestData/"
bazel-bin/cabby/geo/map_processing/map_processor --region Pittsburgh_small --min_s2_level 18 --directory "./cabby/geo/map_processing/poiTestData/"
bazel-bin/cabby/geo/sample_poi --region DC --min_s2_level 18 --directory "./cabby/geo/map_processing/poiTestData/" --path "./cabby/geo/pathData/dc_geo_paths.gpkg" --n_samples 1

# RVS
bazel build cabby/rvs:*
bazel query cabby/rvs:* | xargs bazel test
bazel-bin/cabby/rvs/generate_rvs --rvs_data_path "./cabby/geo/pathData/pittsburgh_geo_paths.gpkg" --save_instruction_path "./cabby/rvs/data/pittsburgh_instructions.json" 

# Wikidata
bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items --region Pittsburgh

# Wikipedia
bazel-bin/cabby/data/wikipedia/extract_wikipedia_items --titles=New_York_Stock_Exchange,Empire_State_Building

# Wikigeo

bazel-bin/cabby/data/extract_wikigeo_contexts --region Pittsburgh_small --path pittsburgh_small.json
bazel-bin/cabby/data/extract_wikigeo_contexts_with_osm --region Pittsburgh_small --save_path pittsburgh_small_with_osm.json --osm_path  "./cabby/geo/map_processing/poiTestData/pittsburgh_small_poi.pkl"
