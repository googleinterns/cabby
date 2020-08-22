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

# Use alias for Direction to keep code cleaner.
Direction = directions.Direction

def describe_meeting_point(
  pivot: Text, goal: Text,
  bearing_pivot_goal: float, distance_pivot_goal: float) -> Text:
  '''Preliminary example template for generating an RVS instruction.
  
  Arguments:
    pivot: The POI used to orient with respect to the goal.
    goal: The POI that is the intended meeting location.
    bearing_pivot_goal: The relative bearing of the goal when heading to the
      pivot from the start point.
    distance_pivot_goal: The distance from the pivot to the goal.
  Returns:
    A string describing the goal location with respect to the reference.
  
  '''
  direction_description = describe_egocentric_direction(
      directions.(bearing_pivot_goal), 
      distance_pivot_goal)

  return f'go to {goal}, which is {direction_description} {pivot}'

def describe_distance(distance_in_km: float) -> Text:
  """Convert a raw distance into a short verbal description."""
  
  # Round to 100 meters.
  rdist = round(distance_in_km, 1)

  if rdist < 0.1:
    return 'not even 100 meters'
  if rdist < 1.0:
    return f'{int(rdist*1000)} meters'
  else:
    return f'{rdist}km'


def describe_egocentric_direction(
  direction: Direction, distance: float) -> Text:
  """Convert a Direction and distance into a verbal description.
  
  This is very rough cut, to be improved considerably.
  """
  dist_descr = describe_distance(distance)
  if direction == Direction.AHEAD:
    return f'{dist_descr} past'
  elif direction == Direction.SLIGHT_RIGHT:
    return f'{dist_descr} up and to the right of'
  elif direction == Direction.RIGHT:
    return f'{dist_descr} to the right of'
  elif direction == Direction.SLIGHT_LEFT:
    return f'{dist_descr} up and to the left of'
  elif direction == Direction.LEFT:
    return f'{dist_descr} to the left of'
  elif direction == Direction.BEHIND:
    return f'{dist_descr} before you get to'
  else:
    raise ValueError(f'Invalid direction type: {direction}')
