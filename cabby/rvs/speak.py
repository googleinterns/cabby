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

'''Library to support producing synthetic RVS instructions.'''

from typing import Text

from cabby.geo import directions

Direction = directions.Direction


def describe_route(pivot_poi: Text, goal_poi: Text) -> Text:
  '''Preliminary example template for generating an RVS instruction.
  
  Arguments:
    pivot_poi: The POI used to orient with respect to the goal.
    goal_poi: The POI that is the intended meeting location.
  Returns:
    A string describing the goal location with respect to the reference.
  
  '''
  return f'go to the {goal_poi} near the {pivot_poi}.'


def speak_egocentric_direction(direction: int, distance: float) -> Text:
  if direction == Direction.AHEAD:
    return f'{distance} km past'
  elif direction == Direction.SLIGHT_RIGHT:
    return f'{distance} km up and to the right of'
  elif direction == Direction.RIGHT:
    return f'{distance} km to the right of'
  elif direction == Direction.SLIGHT_LEFT:
    return f'{distance} km up and to the left of'
  elif direction == Direction.LEFT:
    return f'{distance} km to the left of'
  elif direction == Direction.BEHIND:
    return f'{distance} km before you get to'
  else:
    raise ValueError(f'Invalid direction type: {direction}')
