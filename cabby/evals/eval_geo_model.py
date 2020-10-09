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

from cabby.evals import utils

FLAGS = flags.FLAGS
flags.DEFINE_string("input_file", None,
          "TSV file containing true and predicted co-ordinates.")
flags.mark_flag_as_required("input_file")


def main(argv):
  del argv  # Unused.
  evaluator = utils.Evaluator()
  error_distances = evaluator.get_error_distances(FLAGS.input_file)
  _ = evaluator.compute_metrics(error_distances)


if __name__ == "__main__":
  app.run(main)