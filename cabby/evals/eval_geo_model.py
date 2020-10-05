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

import collections

import numpy as np
from absl import app, flags
from geopy.distance import great_circle

from cabby import logger

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None,
                    "TSV file containing true and predicted co-ordinates.")
flags.mark_flag_as_required("input_file")
# Object for storing each evaluation tuple parsed from the input file.
EvalDataTuple = collections.namedtuple(
    "EvalDataTuple",
    ["example_id", "true_lat", "true_lng", "predicted_lat", "predicted_lng"])
# 20039 kms is half of earth's circumference (max. great circle distance)
_MAX_LOG_HAVERSINE_DIST = np.log(20039 * 1000)  # in meters.
_EPSILON = 1e-5


class Evaluator:
  """Class for evaluating geo models."""

  def __init__(self):
    self.eval_logger = logger.create_logger("rvs_geo_eval.log", "rvs_geo_eval")
    self.eval_logger.info("Starting evaluation.")

  def get_error_distances(self, input_file):
    """Compute error distance in meters between true and predicted coordinates.

			Args:
				input_file: TSV file containing example_id and true and
					predicted co-ordinates. One example per line.
				eval_logger: Logger object.

			Returns:
				Array of distance error - one per example.
    """
    error_distances = []
    total_examples = 0
    for line in open(input_file):
      toks = line.strip().split("\t")
      if len(toks) != 5:
        self.eval_logger.warning("Unexpected line format: [%s]. Skipping", line)
        continue
      eval_tuple = EvalDataTuple(toks[0], float(toks[1]), float(toks[2]),
                                 float(toks[3]), float(toks[4]))
      err = great_circle((eval_tuple.true_lat, eval_tuple.true_lng),
                         (eval_tuple.predicted_lat, eval_tuple.predicted_lng)).m
      error_distances.append(err)
      total_examples += 1
      if total_examples % 100 == 0:
        self.eval_logger.info("Processed [%d]", total_examples)
    self.eval_logger.debug(error_distances)
    return error_distances

  def compute_metrics(self, error_distances):
    """Compute distance error metrics given an array of error distances.

			Args:
				error_distances: Array of distance errors.
				eval_logger: Logger object.
    """
    num_examples = len(error_distances)
    if num_examples == 0:
      self.eval_logger.error("No examples to be evaluated.")
    accuracy = len(np.where(error_distances == 0.)[0]) / num_examples
    mean_distance, median_distance = np.mean(error_distances), np.median(
        error_distances)
    log_distance = np.sort(
        np.log(error_distances + np.ones_like(error_distances) * _EPSILON))
    # AUC for the distance errors curve. Smaller the better.
    auc = np.trapz(log_distance)

    # Normalized AUC by maximum error possible.
    norm_auc = auc / (_MAX_LOG_HAVERSINE_DIST * (num_examples - 1))

    self.eval_logger.info(
        "Metrics: \nExact accuracy : [%.2f]\nmean error [%.2f], " +
        "\nmedian error [%.2f]\nAUC of error curve [%.2f]", accuracy,
        mean_distance, median_distance, norm_auc)


def main(argv):
  del argv  # Unused.
  evaluator = Evaluator()
  error_distances = evaluator.get_error_distances(FLAGS.input_file)
  evaluator.compute_metrics(error_distances)


if __name__ == "__main__":
  app.run(main)