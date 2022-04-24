REGION_NAME="UTAustin"
OUTPUT_DIR=$HOME/tmp/cabby_run/$REGION_NAME
MAP_DIR=$OUTPUT_DIR/map


echo "****************************************"
echo "*                 models               *"
echo "****************************************"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $MAP_DIR


echo "*                 Dual-Encoder-Bert               *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task human --model Dual-Encoder-Bert
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task RVS --model Dual-Encoder-Bert

echo "*                 Classification-Bert               *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task human --model Classification-Bert
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task RVS --model Classification-Bert

echo "*                 S2-Generation-T5               *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task human --model S2-Generation-T5
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5

echo "*                 S2-Generation-T5-Landmarks               *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task human --model S2-Generation-T5-Landmarks
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5-Landmarks

echo "*                 S2-Generation-T5-Path               *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task human --model S2-Generation-T5-Path
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5-Path

echo "*                 S2-Generation-T5-Warmup-start-end               *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task human --model S2-Generation-T5-Warmup-start-end
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR --train_batch_size 8  --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5-Warmup-start-end


echo "****************************************"
echo "*                 Geo                  *"
echo "****************************************"

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
