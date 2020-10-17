REGION_NAME="Pittsburgh_small"
OUTPUT_DIR=$HOME/tmp/cabby_run

echo "****************************************"
echo "*                 Geo                  *"
echo "****************************************"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/poiTestData

# Do we still need geo_computation?
# bazel-bin/cabby/geo/geo_computation --orig_lat 40.749102 --orig_lon -73.984076 -dest_lat 40.748432 --dest_lon -73.982473

bazel-bin/cabby/geo/map_processing/map_processor --region $REGION_NAME --min_s2_level 18 --directory $OUTPUT_DIR/poiTestData/
bazel-bin/cabby/geo/map_processing/map_processor --region DC --min_s2_level 18 --directory $OUTPUT_DIR/poiTestData
bazel-bin/cabby/geo/sample_poi --region DC --min_s2_level 18 --directory $OUTPUT_DIR/poiTestData --path $OUTPUT_DIR/poiTestData/dc_geo_paths.gpkg --n_samples 1

echo "****************************************"
echo "*              Wikidata                *"
echo "****************************************"
bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items --region $REGION_NAME

# Wikipedia
echo "****************************************"
echo "*              Wikipedia               *"
echo "****************************************"
bazel-bin/cabby/data/wikipedia/extract_wikipedia_items --titles=New_York_Stock_Exchange,Empire_State_Building

echo "****************************************"
echo "*              Wikigeo                 *"
echo "****************************************"
bazel-bin/cabby/data/extract_wikigeo_contexts --region $REGION_NAME --output_dir /tmp/wikigeo
bazel-bin/cabby/data/extract_wikigeo_contexts --region $REGION_NAME --output_dir /tmp/wikigeo --osm_path $OUTPUT_DIR/poiTestData/pittsburgh_small_poi.pkl

echo "****************************************"
echo "*                 RVS                  *"
echo "****************************************"
bazel-bin/cabby/rvs/generate_rvs --rvs_data_path $OUTPUT_DIR/poiTestData/dc_geo_paths.gpkg --save_instruction_path $OUTPUT_DIR/dc_small_instructions.json