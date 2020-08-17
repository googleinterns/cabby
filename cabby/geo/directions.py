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
'''Basic functions for working with relative and absolute geo directions.'''

import enum

class Direction(enum.IntEnum):
  AHEAD = 0
  SLIGHT_LEFT = 1
  LEFT = 2
  SLIGHT_RIGHT = 3
  RIGHT = 4
  BEHIND = 5


def angle_in_360(angle: float) -> float:
  if angle < 0:
    return angle + 360
  return angle


def get_egocentric_direction(angle: float) -> int:
  angle = angle_in_360(angle)
  if angle < 30 or angle > 330:
    return Direction.AHEAD
  elif angle < 60:
    return Direction.SLIGHT_RIGHT
  elif angle < 120:
    return Direction.RIGHT
  elif angle < 240:
    return Direction.BEHIND
  elif angle < 300:
    return Direction.LEFT
  else:
    return Direction.SLIGHT_LEFT
