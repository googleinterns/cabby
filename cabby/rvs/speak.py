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

def describe_route(pivot_poi: Text, goal_poi: Text) -> Text:
  '''Preliminary example template for generating an RVS instruction.
  
  Arguments:
    pivot_poi: The POI used to orient with respect to the goal.
    goal_poi: The POI that is the intended meeting location.
  Returns:
    A string describing the goal location with respect to the reference.
  
  '''
  return f'go to the {goal_poi} near the {pivot_poi}.'