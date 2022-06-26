REGION_NAME="UTAustin"
OUTPUT_DIR=$HOME/tmp/cabby_run/$REGION_NAME
MAP_DIR=$OUTPUT_DIR/map

OUTPUT_DIR_MODEL=$HOME/tmp/cabby_run/manhattan
OUTPUT_DIR_MODEL_RVS=$OUTPUT_DIR_MODEL/rvs
OUTPUT_DIR_MODEL_RVS_FIXED_4=$OUTPUT_DIR_MODEL/rvs/fixed_4
OUTPUT_DIR_MODEL_RVS_FIXED_5=$OUTPUT_DIR_MODEL/rvs/fixed_5

OUTPUT_DIR_MODEL_HUMAN=$OUTPUT_DIR_MODEL/human



echo "****************************************"
echo "*                 models               *"
echo "****************************************"
rm -rf $OUTPUT_DIR_MODEL
mkdir -p $OUTPUT_DIR_MODEL
mkdir -p $OUTPUT_DIR_MODEL_RVS
mkdir -p $OUTPUT_DIR_MODEL_RVS_FIXED_4
mkdir -p $OUTPUT_DIR_MODEL_RVS_FIXED_5
mkdir -p $OUTPUT_DIR_MODEL_HUMAN


echo "*                 Dual-Encoder-Bert  - HUMAN DATA             *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model Dual-Encoder-Bert
echo "*                 Dual-Encoder-Bert  - RVS DATA             *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model Dual-Encoder-Bert

echo "*                 Classification-Bert  - HUMAN DATA             *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model Classification-Bert
echo "*                 Classification-Bert  - RVS DATA             *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model Classification-Bert

echo "*                 S2-Generation-T5    - HUMAN DATA           *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model S2-Generation-T5
echo "*                 S2-Generation-T5    - RVS DATA           *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model S2-Generation-T5

echo "*                 S2-Generation-T5-start-text-input    - HUMAN DATA           *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model S2-Generation-T5-start-text-input
echo "*                 S2-Generation-T5-start-text-input    - RVS DATA          *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model S2-Generation-T5-start-text-input

echo "*                 S2-Generation-T5-Landmarks   - HUMAN DATA            *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model S2-Generation-T5-Landmarks
echo "*                 S2-Generation-T5-Landmarks   - RVS DATA            *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model S2-Generation-T5-Landmarks

echo "*                 S2-Generation-T5-Path    - HUMAN DATA           *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model S2-Generation-T5-Path
echo "*                 S2-Generation-T5-Path    - RVS DATA           *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model S2-Generation-T5-Path

echo "*                 S2-Generation-T5-Warmup-start-end   - HUMAN DATA            *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/human --dataset_dir $OUTPUT_DIR_MODEL_HUMAN --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_HUMAN --num_epochs 1 --task human --model S2-Generation-T5-Warmup-start-end

echo "*                 S2-Generation-T5-Warmup-start-end   - RVS DATA            *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS_FIXED_4 --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model S2-Generation-T5-Warmup-start-end --n_fixed_points 4
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS_FIXED_5 --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model S2-Generation-T5-Warmup-start-end --n_fixed_points 5


echo "*                 Text-2-Landmarks-NER-Generation-T5-Warmup   - RVS DATA            *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model Text-2-Landmarks-NER-Generation-T5-Warmup

echo "*                 Landmarks-NER-2-S2-Generation-T5-Warmup   - RVS DATA            *"
bazel-bin/cabby/model/text/model_trainer  --data_dir ~/cabby/cabby/model/text/dataSamples/rvs --dataset_dir $OUTPUT_DIR_MODEL_RVS --region Manhattan --s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1 --task RVS --model Landmarks-NER-2-S2-Generation-T5-Warmup

echo "*                multitask           *"
bazel-bin/cabby/model/text/model_trainer_multitask  --dataset_dir_T5_Warmup_start_end_RVS_fixed_n_5 $OUTPUT_DIR_MODEL_RVS_FIXED_5 --dataset_dir_T5_Warmup_start_end_RVS_fixed_n_4 $OUTPUT_DIR_MODEL_RVS_FIXED_4 --dataset_dir_T5_landmarks_human $OUTPUT_DIR_MODEL_HUMAN --dataset_dir_T5_landmarks_RVS $OUTPUT_DIR_MODEL_RVS --region Manhattan -s2_level 15 --output_dir $OUTPUT_DIR_MODEL_RVS --num_epochs 1

echo "*                Baseline           *"
bazel-bin/cabby/model/baselines --data_dir ~/cabby/cabby/model/text/dataSamples/human  --metrics_dir $OUTPUT_DIR_MODEL_HUMAN  --task human --region Manhattan


echo "****************************************"
echo "*                 Geo                  *"
echo "****************************************"

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $MAP_DIR

bazel-bin/cabby/geo/map_processing/map_processor --region $REGION_NAME --min_s2_level 18 --directory $MAP_DIR
bazel-bin/cabby/geo/sample_poi --region $REGION_NAME --min_s2_level 18 --directory $MAP_DIR --path $MAP_DIR/utaustin_geo_paths.gpkg --n_samples 1

echo "****************************************"
echo "*                 RVS                  *"
echo "****************************************"
bazel-bin/cabby/rvs/generate_rvs --rvs_data_path $MAP_DIR/utaustin_geo_paths.gpkg --save_instruction_dir $OUTPUT_DIR

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
