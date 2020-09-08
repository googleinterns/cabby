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
'''Basic classes and functions for Wikigeo items.'''

from geopandas import GeoDataFrame
import re
from shapely.geometry.point import Point
from typing import Text, Dict

import attr


@attr.s
class RVSPath:
    """Construct a Wikigeo sample.

    `start_point` is the beginning location.
    `end_point` is the goal location.
    `route` is route between the start and end points.
    `main_pivot` is pivot along the route.
    `near_pivot` is the pivot near the goal location. 
    `instruction` is a basic template that includes the points and pivots.      
    """
    start_point: Dict = attr.ib()
    end_point: Dict = attr.ib()
    route: GeoDataFrame = attr.ib()
    main_pivot: Dict = attr.ib()
    near_pivot: Dict = attr.ib()
    instruction: Text = attr.ib(init=False)

    def __attrs_post_init__(self):

        # Creat basic template instruction.
        self.instruction = \
            "Starting at {0} walk past {1} and your goal is {2}, near {3}." \
            .format(self.start_point['name'], self.main_pivot['main_tag'], self.
                    end_point['name'], self.near_pivot['main_tag'])

        # Create centroid point.
        self.main_pivot['centroid'] = self.main_pivot['geometry'] if \
            isinstance(self.main_pivot['geometry'], Point) else \
            self.main_pivot['geometry'].centroid

        self.near_pivot['centroid'] = self.near_pivot['geometry'] if \
            isinstance(self.near_pivot['geometry'], Point) else \
            self.near_pivot['geometry'].centroid

    @classmethod
    def from_points_route_pivots(cls, start, end, route, main_pivot,
                                 near_pivot):
        """Construct an Entity from the start and end points, route, and pivots.
        """
        return RVSPath(
            start,
            end,
            route,
            main_pivot,
            near_pivot
        )
