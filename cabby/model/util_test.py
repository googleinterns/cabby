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
'''Tests for util.py'''
import unittest
import numpy as np
import osmnx as ox
import unittest
from shapely.geometry.point import Point
import sys

from cabby.model import util

class UtilTest(unittest.TestCase):

  def testDistanceProbabilty(self):

    # Distance probability with scale 1 kilometer (1000 meters) and with default
    # shape (2).
    dprob = util.DistanceProbability(1000)

    # Check probability of 900 meters.
    self.assertAlmostEqual(dprob(900), 0.36591269376653923)

    # Distance probabilty with scale 5 km and shape 3.
    dprob5kshape2 = util.DistanceProbability(5000, 3)
    
    # Check probability of 6000 meters.
    self.assertAlmostEqual(dprob5kshape2(6000), 0.21685983257678548)



  def testBinarization(self):
    array_int = np.array([1, 2, 3, 1], dtype=np.int8)    
    representation = util.binary_representation(array_int, 2)
    np.testing.assert_array_equal(
      representation, np.array([[1, 0], [0, 1], [1, 1], [1, 0]]))


if __name__ == "__main__":
  unittest.main()
