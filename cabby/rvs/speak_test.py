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

import unittest

from cabby.geo import directions
from cabby.rvs import speak


class SpeakTest(unittest.TestCase):

  def testSingleOutput(self):
      expected = ('go to Church of the Incarnation, Episcopal, which is 100 '
                  'meters to the left of Graduate Center, CUNY')
      output = speak.describe_meeting_point(
          'Graduate Center, CUNY',
          'Church of the Incarnation, Episcopal',
          -75.84734584372487,
          0.09300668715292916
      )
      self.assertEqual(output, expected)


if __name__ == "__main__":
    unittest.main()
