REGION_NAME="UTAustin"
OUTPUT_DIR=$HOME/tmp/cabby_run/$REGION_NAME
MAP_DIR=$OUTPUT_DIR/map

echo "****************************************"
echo "*                 Geo                  *"
echo "****************************************"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $MAP_DIR

bazel-bin/cabby/geo/map_processing/map_processor --region $REGION_NAME --min_s2_level 18 --directory $MAP_DIR
bazel-bin/cabby/geo/sample_poi --region $REGION_NAME --min_s2_level 18 --directory $MAP_DIR --path $MAP_DIR/utaustin_geo_paths.gpkg --n_samples 1

echo "****************************************"
echo "*              Wikidata                *"
echo "****************************************"
bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items --region $REGION_NAME

echo "****************************************"
echo "*              Wikipedia               *"
echo "****************************************"
bazel-bin/cabby/data/wikipedia/extract_wikipedia_items --titles=New_York_Stock_Exchange,Empire_State_Building

echo "****************************************"
echo "*              Wikigeo                 *"
echo "****************************************"
bazel-bin/cabby/data/create_wikigeo_dataset --region $REGION_NAME --output_dir $OUTPUT_DIR/wikigeo
bazel-bin/cabby/data/create_wikigeo_dataset --region $REGION_NAME --output_dir $OUTPUT_DIR/wikigeo --osm_path $MAP_DIR/utaustin_poi.pkl

echo "****************************************"
echo "*                 RVS                  *"
echo "****************************************"
bazel-bin/cabby/rvs/generate_rvs --rvs_data_path $MAP_DIR/utaustin_geo_paths.gpkg --save_instruction_dir $OUTPUT_DIR