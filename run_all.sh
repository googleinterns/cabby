# Check the types in everything.
pytype cabby

# RVS
bazel build cabby/rvs:*
bazel query cabby/rvs:* | xargs bazel test
bazel-bin/cabby/rvs/generate_rvs --start_lat 40.753628 --start_lon -73.985085 --goal_lat 40.748432 --goal_lon -73.98247

# Geo
bazel-bin/cabby/geo/geo_computation --orig_lat 40.749102 --orig_lon -73.984076 -dest_lat 40.748432 --dest_lon -73.982473
bazel-bin/cabby/geo/map_processing/map_processor --region Pittsburgh --level 18 --directory "/mnt/hackney/data/cabby/poi/temp/"
bazel-bin/cabby/geo/map_processing/map_processor --region Manhattan --level 18 --directory "/mnt/hackney/data/cabby/poi/temp/"
bazel-bin/cabby/geo/sample_poi --region Manhattan --level 18 --directory "/mnt/hackney/data/cabby/poi/v1/"

# Wikidata
bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items --region Pittsburgh

# Wikipedia
bazel-bin/cabby/data/wikipedia/extract_wikipedia_items --titles=New_York_Stock_Exchange,Empire_State_Building

# Wikigeo
bazel-bin/cabby/data/extract_wikigeo_contexts --region Bologna --path bologna.json