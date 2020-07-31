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

'''Tests for speak.py'''

from geo import walk

import unittest
from shapely.geometry.point import Point

class WalkTest(unittest.TestCase):

  def testSingleOutput(self):
    start_point=(40.748432,-73.982473)
    end_point=(40.74907,-73.98403)
    list_points = walk.compute_route(start_point,end_point)
    eps=0.0005
    for point in list_points:
      self.assertLessEqual(point.y,end_point[0]+eps)
      self.assertGreaterEqual(point.y,start_point[0]-eps)
      self.assertLessEqual(point.x,start_point[1]+eps)
      self.assertGreaterEqual(point.x,end_point[1]-eps)

if __name__ == "__main__":
    unittest.main()

