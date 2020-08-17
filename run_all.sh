# Check the types in everything.
pytype cabby

# RVS
bazel build cabby/rvs:*
bazel query cabby/rvs:* | xargs bazel test
bazel-bin/cabby/rvs/generate_rvs --ref_poi "Empire State Building" --goal_poi "pharmacy"

# Geo
bazel-bin/cabby/geo/geo_computation --orig_lat 40.749102 --orig_lon -73.984076 -dest_lat 40.748432 --dest_lon -73.982473
bazel-bin/cabby/geo/map_processing/map_processor --region Pittsburgh --level 18
bazel-bin/cabby/geo/map_processing/map_processor --region Manhattan --level 18

# Wikidata
bazel test cabby/data/wikidata:query_test
bazel build cabby/data/wikidata/extract_geofenced_wikidata_items
bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items --region Pittsburgh


