# Copyright 2020 The Flax Authors.
#
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

"""Dual-encoder framework for text and S2Cellids matching.

Example command line call:
$ bazel-bin/cabby/model/text/dual_encoder/model_trainer \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dual_encoder/dataset/pittsburgh \
  --region Pittsburgh \ 
  --s2_level 12 \
  --output_dir ~/tmp/output\

"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import os 
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/tzuf_google_com/dev/cabby")


from cabby.model.text.dual_encoder import train
from cabby.model.text.dual_encoder import dataset
from cabby.model.text.dual_encoder import model

# FLAGS = flags.FLAGS


# flags.DEFINE_string("data_dir", None,
#           "The directory from which to load the dataset.")
# flags.DEFINE_string("dataset_dir", None,
#           "The directory to save\load dataloader.")
# flags.DEFINE_enum(
#   "region", None, ['Pittsburgh', 'Manhattan'],
#   "Map areas: Manhattan or Pittsburgh.")
# flags.DEFINE_integer("s2_level", None, "S2 level of the S2Cells.")
# flags.DEFINE_string("output_dir", None,
#           "The directory where the model and results will be save to.")

# flags.DEFINE_float(
#   'learning_rate', default=5e-5,
#   help=('The learning rate for the Adam optimizer.'))

# flags.DEFINE_integer(
#   'train_batch_size', default=16,
#   help=('Batch size for training.'))

# flags.DEFINE_integer(
#   'test_batch_size', default=16,
#   help=('Batch size for testing and validating.'))

# flags.DEFINE_integer(
#   'num_epochs', default=5,
#   help=('Number of training epochs.'))

# flags.DEFINE_integer(
#   'eval_every', None,
#   help=('Evaluation after a number of trained samples.'))

# # Required flags.
# flags.mark_flag_as_required("data_dir")
# flags.mark_flag_as_required("dataset_dir")
# flags.mark_flag_as_required("region")
# flags.mark_flag_as_required("s2_level")
# flags.mark_flag_as_required("output_dir")

class FLAGS1:
  def __init__(self):
    self.data_dir = "/home/tzuf_google_com/data/wikigeo/pittsburgh"
    self.dataset_dir = "/home/tzuf_google_com/model/dual_encoder/dataset/pittsburgh"
    self.region = "Pittsburgh"
    self.s2_level = 12 
    self.output_dir =  "/home/tzuf_google_com/tmp/output/dual"
    self.learning_rate = 5e-5
    self.test_batch_size = 4
    self.train_batch_size = 4
    self.num_epochs = 2


FLAGS= FLAGS1()
# def main(argv):
def temp():

  if not os.path.exists(FLAGS.dataset_dir):
    sys.exit("Dataset path doesn't exsist: {}.".format(FLAGS.dataset_dir))

  dataset_path = os.path.join(FLAGS.dataset_dir, str(FLAGS.s2_level))
  train_path_dataset = os.path.join(dataset_path,'train.pth')
  valid_path_dataset = os.path.join(dataset_path,'valid.pth')
  lables_dictionary = os.path.join(dataset_path,"label_to_cellid.npy")


  if os.path.exists(dataset_path):
    logging.info("Loading dataset from <== {}.".format(dataset_path))
    train_dataset = torch.load(train_path_dataset)
    valid_dataset = torch.load(valid_path_dataset)
    label_to_cellid = np.load(lables_dictionary, allow_pickle='TRUE').item()
    n_lables = len(label_to_cellid)

  else:
    logging.info("Preparing data.")
    train_dataset, valid_dataset, test_dataset, label_to_cellid = dataset.create_dataset(
      FLAGS.data_dir, FLAGS.region, FLAGS.s2_level)
    n_lables = len(label_to_cellid)
    logging.info("Number of lables: {}".format(n_lables))

    
    # Save to dataloaders and lables to cells dictionary.
    os.mkdir(dataset_path)
    torch.save(train_dataset, train_path_dataset)
    torch.save(valid_dataset, valid_path_dataset)
    np.save(lables_dictionary, label_to_cellid) 
    logging.info("Saved data to ==> {}.".format(dataset_path))

  train_loader = DataLoader(
  train_dataset, batch_size=FLAGS.train_batch_size, shuffle=True)
  valid_loader = DataLoader(
  valid_dataset, batch_size=FLAGS.test_batch_size, shuffle=False)

  device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

  dual_encoder = model.DualEncoder()
  if torch.cuda.device_count() > 1:
    logging.info("Using {} GPUs.".format(torch.cuda.device_count()))
    dual_encoder = nn.DataParallel(dual_encoder)

  dual_encoder.to(device)

  optimizer = torch.optim.Adam(dual_encoder.parameters(), lr=FLAGS.learning_rate)

  train.train_model(
    model=dual_encoder,
    device=device,
    num_epochs=FLAGS.num_epochs,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    label_to_cellid = label_to_cellid,
    file_path=FLAGS.output_dir, 
    )

temp()

# if __name__ == '__main__':
#   app.run(main)



