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
'''Tests for RVS model evaluation utils.py'''

import unittest

from cabby.evals import utils


class UtilsTest(unittest.TestCase):

  def setUp(self):
    self.evaluator = utils.Evaluator()
    self.test_eval_input = "cabby/evals/testdata/sample_test_evals.tsv"

  def assertListAlmostEqual(self, list1, list2, tol=2):
    self.assertEqual(len(list1), len(list2))
    for a, b in zip(list1, list2):
      self.assertAlmostEqual(a, b, tol)

  def testErrorDistances(self):
    actual_errors = self.evaluator.get_error_distances(
      self.test_eval_input)
    self.assertListAlmostEqual(
      actual_errors, [62259.67, 58923.88, 0.0, 1854892.25, 69314.61])

  def testComputeMetrics(self):
    actual_metrics = self.evaluator.compute_metrics([10, 50, 20, 0, 30])
    self.assertAlmostEqual(actual_metrics.accuracy, 0.2, 2)
    self.assertAlmostEqual(actual_metrics.mean_distance, 22.0, 2)
    self.assertAlmostEqual(actual_metrics.median_distance, 20.0, 2)
    self.assertAlmostEqual(actual_metrics.max_error, 50.0, 2)
    self.assertAlmostEqual(actual_metrics.norm_auc, 0.07, 2)


if __name__ == "__main__":
  unittest.main()
