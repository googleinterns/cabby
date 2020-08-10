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