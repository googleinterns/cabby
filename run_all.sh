# Check the types in everything.
pytype cabby

# Build, test and run the cabby.rvs subpackage.
bazel build cabby/rvs:*
bazel query cabby/rvs:* | xargs bazel test
bazel-bin/cabby/rvs/generate_rvs --ref_poi "Empire State Building" --goal_poi "pharmacy"

# Build, test and run the cabby.geo subpackage.
bazel build cabby/geo:*
bazel query cabby/geo:* | xargs bazel test
bazel-bin/cabby/geo/geo_computation --orig_lat 40.749102 --orig_lon -73.984076 -dest_lat 40.748432 --dest_lon -73.982473
bazel-bin/cabby/geo/map_processing/map_processor --region Pittsburgh --level 18

# This isn't working.
# bazel-bin/cabby/geo/map_processing/map_processor --region Manhattan --level 18

# Build, test and run the cabby.geo subpackage.
bazel test cabby/data/wikidata:query_test
bazel build cabby/data/wikidata/extract_geofenced_wikidata_items
bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items --region Pittsburgh

# Build, test and run the cabby.data.wikidata subpackage.

bazel test cabby/data/wikidata:query_test
bazel build cabby/data/wikidata/extract_geofenced_wikidata_items
bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items --region Pittsburgh

# Build, test and run the cabby.data.wikipedia subpackage.

bazel test cabby/data/wikipedia:query_test
bazel build cabby/data/wikipedia/extract_wikipedia_items
bazel-bin/cabby/data/wikipedia/extract_wikipedia_items --titles=New_York_Stock_Exchange,Empire_State_Building

# Build, test and run the cabby.data subpackage.

bazel test cabby/data:extract_test
bazel build cabby/data/save
bazel-bin/cabby/data/save --region Pittsburgh --path geodata.txt
