# coding=utf-8
# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility for distance based evaluation for geo models in RVS.

Example:
$ bazel run -- cabby/evals/eval_geo_model \
  --input_file /mnt/hackney/data/cabby/evals/sample_predictions.tsv
The input_file is expected to be a tab-separated file with:
  example_id
  true_latitude
  true_longitude
  predicted_latitude
  predicted_longitude
Any lines which do not comply to this format will be skipped.
"""

from absl import app, flags

import torch

from cabby import logger
from cabby.evals import utils
from cabby.model.text.s2cellid_prediction import train

FLAGS = flags.FLAGS
flags.DEFINE_bool("preprocess_pytorch_output", False,
          "If true, reads input as a pytorch file and generates "
          "intermediate TSV.")
flags.DEFINE_string("input_file", None,
          "TSV file containing true and predicted co-ordinates.")
flags.mark_flag_as_required("input_file")


def main(argv):
  del argv  # Unused
  eval_logger = logger.create_logger("rvs_geo_eval.log", "rvs_geo_eval")
  tsv_input = FLAGS.input_file
  if FLAGS.preprocess_pytorch_output:
    eval_logger.info("Preprocessing the pytorch output")
    state_dict = train.load_metrics(
      FLAGS.input_file, torch.device("cpu", 0))
    eval_logger.info("Keys in state dictionary : [%s]", state_dict.keys())

    output_file = FLAGS.input_file + ".raw_data.tsv"
    fout = open(output_file, "w")

    for cnt, (true, pred) in enumerate(zip(state_dict['true_points_list'][0], state_dict['pred_points_list'][0])):
      fout.write("%d\t%f\t%f\t%f\t%f\n" %
             (cnt, true[0], true[1], pred[0], pred[1]))
      if cnt % 1000:
        eval_logger.info("Read [%d] examples", cnt)
    fout.close()
    eval_logger.info(
      "Intermediate raw data TSV is generated in : [%s]", output_file)
    tsv_input = output_file

  evaluator = utils.Evaluator()
  error_distances = evaluator.get_error_distances(tsv_input)
  evaluator.compute_metrics(error_distances)


if __name__ == "__main__":
  app.run(main)
