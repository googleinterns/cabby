
echo "**          S2-Generation-T5-Landmarks - RVS         **"
bazel-bin/cabby/model/text/model_trainer  --data_dir /mnt/hackney/rvs/model/manhattan --dataset_dir /mnt/hackney/rvs/model/manhattan/fixedlandmarks --region Manhattan --s2_level 18 --output_dir /mnt/hackney/rvs/multitask/18  --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5-Landmarks --is_single_sample_train True
   
for i in {1..1000}
do
   echo "~~~~~~~~~~~~~~~~~~~~~~~~          Epoch $i         ~~~~~~~~~~~~~~~~~~~~~~~~"
   echo "**          S2-Generation-T5-Landmarks - RVS         **"
   bazel-bin/cabby/model/text/model_trainer  --data_dir /mnt/hackney/rvs/model/manhattan --dataset_dir /mnt/hackney/rvs/model/manhattan/fixedlandmarks --region Manhattan --s2_level 18 --output_dir /mnt/hackney/rvs/multitask/18 --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5-Landmarks --model_path /mnt/hackney/rvs/multitask/18/model.pt --is_val_loss_from_model True --is_single_sample_train True
   echo "**          S2-Generation-T5-Landmarks - human         **"
   bazel-bin/cabby/model/text/model_trainer  --data_dir /mnt/hackney/human_data/landmarks_data --dataset_dir /mnt/hackney/human_data/fixedlandmarks  --region Manhattan --s2_level 18 --output_dir /mnt/hackney/rvs/multitask/18 --test_batch_size 64 --num_epochs 1 --task human --model S2-Generation-T5-Landmarks --model_path /mnt/hackney/rvs/multitask/18/model.pt --is_val_loss_from_model True --is_single_sample_train True
   echo "**          S2-Generation-T5-Warmup-start-end - RVS         **"
   bazel-bin/cabby/model/text/model_trainer  --data_dir /mnt/hackney/rvs/data --dataset_dir /mnt/hackney/rvs/model/manhattan/warmup-start-end --region Manhattan --s2_level 18 --output_dir /mnt/hackney/rvs/multitask/18 --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5-Warmup-start-end --model_path /mnt/hackney/rvs/multitask/18/model.pt --is_val_loss_from_model True --is_single_sample_train True
done
echo "**          S2-Generation-T5-Landmarks - RVS - final        **"
bazel-bin/cabby/model/text/model_trainer  --data_dir /mnt/hackney/rvs/model/manhattan --dataset_dir /mnt/hackney/rvs/model/manhattan/fixedlandmarks --region Manhattan --s2_level 18 --output_dir /mnt/hackney/rvs/multitask/18 --test_batch_size 64 --num_epochs 1 --task RVS --model S2-Generation-T5-Landmarks --model_path /mnt/hackney/rvs/multitask/18/model.pt --is_val_loss_from_model True --is_single_sample_train True
echo "**          S2-Generation-T5-Landmarks - human - final      **"
bazel-bin/cabby/model/text/model_trainer  --data_dir /mnt/hackney/human_data/landmarks_data --dataset_dir /mnt/hackney/human_data/fixedlandmarks  --region Manhattan --s2_level 18 --output_dir /mnt/hackney/rvs/multitask/18  --train_batch_size 16  --test_batch_size 64 --num_epochs 1 --task human --model S2-Generation-T5-Landmarks --model_path /mnt/hackney/rvs/multitask/18/model.pt --is_val_loss_from_model True