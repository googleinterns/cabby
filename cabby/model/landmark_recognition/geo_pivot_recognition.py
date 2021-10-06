# coding=utf-8
# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual-encoder framework for text and S2Cellids matching with pivot 
recognition in process.

Example command line call:
$ bazel-bin/cabby/model/text/dual_encoder/geo_pivot_recognition \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dual_encoder/dataset/pittsburgh \
  --region Pittsburgh \ 
  --s2_level 12 \
  --output_dir ~/tmp/output/dual\
  --train_batch_size 32 \
  --test_batch_size 32 \

For infer:
$ bazel-bin/cabby/model/text/dual_encoder/geo_pivot_recognition \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dual_encoder/dataset/pittsburgh \
  --region Pittsburgh \
  --s2_level 12 \
  --test_batch_size 32 \
  --infer_only True \
  --model_path ~/tmp/model/dual \
  --output_dir ~/tmp/output/dual\



"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import os 
import sys
from scipy.sparse.construct import random
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertForTokenClassification
from transformers import AdamW, pipeline, get_linear_schedule_with_warmup
import random



sys.path.append("/home/onlp_gcp_biu/cabby")
from cabby.evals import utils as eu
from cabby.model.text.dual_encoder import run as run_geo
from cabby.model.text.dual_encoder import dataset_rvs
from cabby.model.text.dual_encoder import model
from cabby.model.text.dual_encoder import dataset_item
from cabby.model.landmark_recognition import dataset_bert as dataset_bert_pivot
from cabby.model.landmark_recognition import run as run_pivot
from cabby.model.text import util
from cabby.geo import regions
from cabby.geo import walk

# FLAGS = flags.FLAGS

landmark_list = walk.LANDMARK_TYPES + [dataset_bert_pivot.EXTRACT_ALL_PIVOTS]
landmark_msg = "Landmark type: " + ','.join(landmark_list)

# flags.DEFINE_string("data_dir", None,
#           "The directory from which to load the dataset.")
# flags.DEFINE_string("dataset_dir", None,
#           "The directory to save\load dataloader.")
# flags.DEFINE_enum(
#   "region", None, regions.SUPPORTED_REGION_NAMES, 
#   regions.REGION_SUPPORT_MESSAGE)
# flags.DEFINE_integer("s2_level", None, "S2 level of the S2Cells.")
# flags.DEFINE_string("output_dir", None,
#           "The directory where the model and results will be save to.")
# flags.DEFINE_float(
#   'learning_rate', default=5e-5,
#   help=('The learning rate for the Adam optimizer.'))

# flags.DEFINE_string("model_path", None,
#           "A path of a model the model to be fine tuned\ evaluated.")


# flags.DEFINE_enum(
#   "pivot_type", None, landmark_list,
#   landmark_msg)

# flags.DEFINE_integer(
#   'train_batch_size', default=4,
#   help=('Batch size for training.'))

# flags.DEFINE_integer(
#   'test_batch_size', default=4,
#   help=('Batch size for testing and validating.'))

# flags.DEFINE_integer(
#   'num_epochs', default=5,
#   help=('Number of training epochs.'))

# flags.DEFINE_bool(
#   'infer_only', default=False,
#   help=('Train and infer\ just infer.'))

# flags.DEFINE_bool(
#   'is_distance_distribution', default=False,
#   help=(
#     'Add probability over cells according to the distance from start point.'+ 
#     'This is optional only for RVS and RUN.'))


# # Required flags.
# flags.mark_flag_as_required("data_dir")
# flags.mark_flag_as_required("dataset_dir")
# flags.mark_flag_as_required("region")
# flags.mark_flag_as_required("s2_level")

import pandas as pd
FLAGS = pd.DataFrame({})
FLAGS.data_dir = "/home/onlp_gcp_biu/cabby/cabby/rvs/sample_instructions/manhattan/"
FLAGS.dataset_dir = "/home/onlp_gcp_biu/tmp/output/model/datasets/geo_pivot_recognition"
FLAGS.region = "Manhattan"
FLAGS.s2_level = 18
FLAGS.infer_only = False
FLAGS.train_batch_size = 4
FLAGS.test_batch_size = 16
FLAGS.model_path = None #"/home/onlp_gcp_biu/tmp/output/model/geo_pivot_recognition.pt"
FLAGS.learning_rate = 0.001
FLAGS.num_epochs = 2
FLAGS.output_dir = "/home/onlp_gcp_biu/tmp/output/model/"
FLAGS.is_distance_distribution = False
FLAGS.pivot_type = 'main_pivot'
FLAGS.max_grad_norm = 1.0

# def main(argv):
def main():

  if not os.path.exists(FLAGS.dataset_dir):
    sys.exit("Dataset path doesn't exist: {}.".format(FLAGS.dataset_dir))

  dataset_path = os.path.join(FLAGS.dataset_dir, str(FLAGS.s2_level))
  train_path_dataset = os.path.join(dataset_path,'train.pth')
  valid_path_dataset = os.path.join(dataset_path,'valid.pth')
  test_path_dataset = os.path.join(dataset_path,'test.pth')
  unique_cellid_path = os.path.join(dataset_path,"unique_cellid.npy")
  tensor_cellid_path = os.path.join(dataset_path,"tensor_cellid.pth")
  label_to_cellid_path = os.path.join(dataset_path,"label_to_cellid.npy")

 
  if os.path.exists(dataset_path):
    dataset_text = dataset_item.TextGeoDataset.load(
      dataset_path = dataset_path, 
      train_path_dataset = train_path_dataset, 
      valid_path_dataset = valid_path_dataset, 
      test_path_dataset = test_path_dataset, 
      label_to_cellid_path = label_to_cellid_path, 
      unique_cellid_path = unique_cellid_path, 
      tensor_cellid_path = tensor_cellid_path)

  else:
    logging.info("Preparing data.")
    dataset_text = dataset_rvs.create_dataset(
            data_dir = FLAGS.data_dir, 
            region = FLAGS.region, 
            s2level = FLAGS.s2_level, 
            infer_only= FLAGS.infer_only,
            is_prob=FLAGS.is_distance_distribution
    )
    

    dataset_item.TextGeoDataset.save(
      dataset_text = dataset_text,
      dataset_path = dataset_path, 
      train_path_dataset = train_path_dataset, 
      valid_path_dataset = valid_path_dataset, 
      test_path_dataset = test_path_dataset, 
      label_to_cellid_path = label_to_cellid_path, 
      unique_cellid_path = unique_cellid_path, 
      tensor_cellid_path = tensor_cellid_path)


  ds_train_pivot, ds_val_pivot, ds_test_pivot = dataset_bert_pivot.create_dataset(
      FLAGS.data_dir, FLAGS.region, FLAGS.s2_level, FLAGS.pivot_type)

  padSequence = dataset_bert_pivot.PadSequence()

  train_pivot_dataloader = DataLoader(
      ds_train_pivot, batch_size=FLAGS.train_batch_size, collate_fn=padSequence)
  val_pivot_dataloader = DataLoader(
      ds_val_pivot, batch_size=FLAGS.test_batch_size, collate_fn=padSequence)
  test_pivot_dataloader = DataLoader(
      ds_test_pivot, batch_size=FLAGS.test_batch_size, collate_fn=padSequence)


  logging.info("Number of unique cells: {}".format(
      len(dataset_text.unique_cellids)))

  train_loader = None
  valid_loader = None
  if FLAGS.infer_only == False:
    train_loader_geo = DataLoader(
      dataset_text.train, batch_size=FLAGS.train_batch_size, shuffle=False)
    valid_loader_geo = DataLoader(
      dataset_text.valid, batch_size=FLAGS.test_batch_size, shuffle=False)
  test_loader_geo = DataLoader(
    dataset_text.test, batch_size=FLAGS.test_batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


  dual_encoder = model.DualEncoder(output_dim= 200, output_landmark=100)
  if FLAGS.model_path is not None:
    if not os.path.exists(FLAGS.model_path):
      sys.exit(f"The model's path does not exists: {FLAGS.model_path}")
    util.load_checkpoint(
      load_path=FLAGS.model_path, model=dual_encoder, device=device)
  if torch.cuda.device_count() > 1:
    logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
    dual_encoder = nn.DataParallel(dual_encoder)

  dual_encoder.to(device)

  optimizer_geo = torch.optim.Adam(
    dual_encoder.parameters(), lr=FLAGS.learning_rate)
  
  model_pivot = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=True
  )

  model_pivot.to(device)

  param_optimizer_pivot = list(model_pivot.named_parameters())
  no_decay = ['bias', 'gamma', 'beta']
  optimizer_grouped_parameters_pivot = [
    {'params': [p for n, p in param_optimizer_pivot if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer_pivot if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
  ]

  optimizer_pivot = AdamW(
    optimizer_grouped_parameters_pivot,
    lr=3e-5,
    eps=1e-8
  )

  total_steps_pivot = len(train_pivot_dataloader) * FLAGS.num_epochs

  scheduler_pivot = get_linear_schedule_with_warmup(
    optimizer_pivot,
    num_warmup_steps=0,
    num_training_steps=total_steps_pivot
  )



  trainer_geo = run_geo.Trainer(
    model=dual_encoder,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer_geo,
    train_loader=train_loader_geo,
    valid_loader=valid_loader_geo,
    test_loader=test_loader_geo,
    unique_cells = dataset_text.unique_cellids,
    file_path=FLAGS.output_dir, 
    cells_tensor = dataset_text.unique_cellids_binary,
    label_to_cellid = dataset_text.label_to_cellid,
    is_distance_distribution = FLAGS.is_distance_distribution
    )

  for epoch_idx in range(FLAGS.num_epochs):
    running_loss = 0.0
    for batch_idx , batch in enumerate(zip(train_pivot_dataloader, train_loader_geo)):
      batch_pivot, batch_geo = batch
      outputs_pivot = run_pivot.run_single(batch_pivot, model_pivot, device)
      loss_pivot = outputs_pivot.loss
      hidden_states_pivot = outputs_pivot.hidden_states[-1][:,-1,:]

      train_pivot = random.randint(0,1)
      if train_pivot:
        loss_pivot.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model_pivot.parameters(), max_norm=FLAGS.max_grad_norm)
        
        # Update parameters.
        optimizer_pivot.step()

        # Update the learning rate.
        scheduler_pivot.step()

      else:
        trainer_geo.optimizer.zero_grad()
        text_geo = {key: val.to(device) for key, val in batch_geo['text'].items()}
        cellids_geo = batch_geo['cellid'].float().to(device)
        neighbor_cells_geo = batch_geo['neighbor_cells'].float().to(device) 
        far_cells_geo = batch_geo['far_cells'].float().to(device)
        (
        text_embedding, 
        cellid_embedding, 
        cellid_embedding_neighbor, 
        cellid_embedding_far) = trainer_geo.compute_embed(
          text_geo, cellids_geo, neighbor_cells_geo, far_cells_geo)
        
        landmark_encode = dual_encoder.landmark_main(hidden_states_pivot)
        full_text_representation = torch.cat((text_embedding, landmark_encode), axis=-1)
        target_positive, target_negative = trainer_geo.get_targets(cellids_geo.shape[0])

        loss_cellid, loss_neighbor, loss_far = trainer_geo.compute_loss_raw(
          full_text_representation, 
          cellid_embedding, 
          cellid_embedding_neighbor,
          cellid_embedding_far,
          target_positive, 
          target_negative)
        
        loss_geo = (loss_cellid + loss_neighbor + loss_far).mean()

        loss_geo.backward()
        trainer_geo.optimizer.step()

        # Update running values.
        running_loss += loss_geo.item()


    print (f"Epoch {epoch_idx}, Loss: {running_loss/batch_idx}")
  
    # Evaluation step.
    predictions, true_vals, _, _ = trainer_geo.evaluate_landmark(
      model_pivot, valid_loader_geo, val_pivot_dataloader)

    accuracy = accuracy_score(true_vals, predictions)

    logging.info(f'Epoch: {epoch_idx}/{FLAGS.num_epochs}, accuracy: {accuracy}')
      

    #     def train_single(self):
    # self.optimizer.zero_grad()
    # text = {key: val.to(self.device) for key, val in batch['text'].items()}
    # cellids = batch['cellid'].float().to(self.device)
    # neighbor_cells = batch['neighbor_cells'].float().to(self.device) 
    # far_cells = batch['far_cells'].float().to(self.device)

    # loss = self.compute_loss(text, cellids, 
    #   neighbor_cells, far_cells)
    
    # return loss
    



  # if FLAGS.infer_only:
  #   logging.info("Starting to infer model.")
  #   valid_loss, predictions, true_vals, true_points, pred_points = (
  #     trainer.evaluate(validation_set = False))

  #   util.save_metrics_last_only(
  #     trainer.metrics_path, 
  #     true_points, 
  #     pred_points)

  #   accuracy = accuracy_score(true_vals, predictions)

  #   evaluator = eu.Evaluator()
  #   error_distances = evaluator.get_error_distances(trainer.metrics_path)
  #   _, mean_distance, median_distance, max_error, norm_auc = (
  #     evaluator.compute_metrics(error_distances))

  #   logging.info(f"\nTest Accuracy: {accuracy}, \n" +
  #   f"Mean distance: {mean_distance},\nMedian distance: {median_distance},\n" +
  #   f"Max error: {max_error},\nNorm AUC: {norm_auc}")

  # else: 
  #   logging.info("Starting to train model.")
  #   trainer.train_model()

  print("END")  


main()
# if __name__ == '__main__':
#   app.run(main)



