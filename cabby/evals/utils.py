# # coding=utf-8
# # Copyright 2020 Google LLC
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #   http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """Library for evaluation functions for RVS models."""

# import collections

# import numpy as np
# from geopy.distance import great_circle

# from cabby import logger

# # Object for storing each evaluation tuple parsed from the input file.
# EvalDataTuple = collections.namedtuple(
#   "EvalDataTuple",
#   ["example_id", "true_lat", "true_lng", "predicted_lat", "predicted_lng"])
# # Object for evaluation metrics.
# EvalMetrics = collections.namedtuple(
#   "EvalMetrics",
#   ["accuracy", "mean_distance", "median_distance", "max_error", "norm_auc"])
# # 20039 kms is half of earth's circumference (max. great circle distance)
# _MAX_LOG_HAVERSINE_DIST = np.log(20039 * 1000)  # in meters.
# _EPSILON = 1e-5


# class Evaluator:
#   """Class for evaluating geo models."""

#   def __init__(self):
#     self.eval_logger = logger.create_logger(
#       "rvs_geo_eval.log", "rvs_geo_eval")
#     self.eval_logger.info("Starting evaluation.")

#   def get_error_distances(self, input_file):
#     """Compute error distance in meters between true and predicted coordinates.

#         Args:
#         input_file: TSV file containing example_id and true and
#           predicted co-ordinates. One example per line.
#         eval_logger: Logger object.

#         Returns:
#         Array of distance error - one per example.
#     """
#     error_distances = []
#     total_examples = 0
#     for line in open(input_file):
#       toks = line.strip().split("\t")
#       if len(toks) != 5:
#         self.eval_logger.warning(
#           "Unexpected line format: [%s]. Skipping", line)
#         continue
#       eval_tuple = EvalDataTuple(toks[0], float(toks[1]), float(toks[2]),
#                      float(toks[3]), float(toks[4]))
#       err = great_circle((eval_tuple.true_lat, eval_tuple.true_lng),
#                  (eval_tuple.predicted_lat, eval_tuple.predicted_lng)).m
#       error_distances.append(err)
#       total_examples += 1
#       if total_examples % 100 == 0:
#         self.eval_logger.info("Processed [%d]", total_examples)
#     return error_distances

#   def compute_metrics(self, error_distances):
#     """Compute distance error metrics given an array of error distances.

#         Args:
#         error_distances: Array of distance errors.
#         eval_logger: Logger object.
#     """
#     num_examples = len(error_distances)
#     if num_examples == 0:
#       self.eval_logger.error("No examples to be evaluated.")
#     accuracy = float(
#       len(np.where(np.array(error_distances) == 0.)[0])) / num_examples
#     mean_distance, median_distance, max_error = np.mean(error_distances), np.median(
#       error_distances), np.max(error_distances)
#     log_distance = np.sort(
#       np.log(error_distances + np.ones_like(error_distances) * _EPSILON))
#     # AUC for the distance errors curve. Smaller the better.
#     auc = np.trapz(log_distance)

#     # Normalized AUC by maximum error possible.
#     norm_auc = auc / (_MAX_LOG_HAVERSINE_DIST * (num_examples - 1))

#     self.eval_logger.info(
#       "Metrics: \nExact accuracy : [%.2f]\nmean error [%.2f], " +
#       "\nmedian error [%.2f]\nmax error [%.2f]\n" +
#       "AUC of error curve [%.2f]", accuracy,
#       mean_distance, median_distance, max_error, norm_auc)
#     return EvalMetrics(accuracy,
#                mean_distance, median_distance, max_error, norm_auc)
