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
'''Basic classes and functions for Wikipedia items.'''

import re
from typing import Text

import attr
from shapely.geometry.point import Point

def points_from_coord(coords):
    return Point(coords[1], coords[0])

@attr.s
class GeoEntity:
    """Simplifed representation of a geo entity.

    `start_point` is the starting point.
    `end_point` is the end point.
    `main_pivot` is the pivot point along the route.
    `near_pivot` is the pivot point near the goal.
    `instruction` is the template instruction that includes the pivots.
    """
    start_point: Point = attr.ib()
    end_point: Point = attr.ib()
    main_pivot: Point = attr.ib()
    near_pivot: Point = attr.ib()
    instruction: Text = attr.ib()

    @classmethod
    def from_api_result(cls, result):
        """Construct an Entity from the results of the Wikimedia API query."""
        return GeoEntity(
            points_from_coord(result['start']),
            points_from_coord(result['end']),
            points_from_coord(result['main_pivot']),
            points_from_coord(result['near_pivot']),
            result['instruction']
        )

