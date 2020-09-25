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

from cabby.geo import util


VERSION = 0.1


@attr.s
class RVSData:
    """Construct a RVSdata sample.
    `start_point` is the beginning location.
    `end_point` is the goal location.
    `distance` path distance between start and end point.
    `instructions` the instructions describing how to get to the end point.
    `id` is the id of the entity.
    `sample` RVS sample.
    """
    start_point: tuple = attr.ib()
    end_point: tuple = attr.ib()
    start_osmid: int = attr.ib()
    end_osmid: int = attr.ib()
    distance: int = attr.ib()
    instructions: Text = attr.ib()
    id: int = attr.ib()
    sample: dict = attr.ib(init=False)

    def __attrs_post_init__(self):
        # Create RVS sample
        self.sample = {
            'start_point': self.start_point, 
            'start_osmid': self.start_osmid,
            'end_point': self.end_point, 
            'end_osmid': self.start_osmid,
            'distance':self.distance, 
            'instructions': self.instructions, 
            'id': self.id, 
            'version': VERSION
        }

    @classmethod
    def from_geo_entities(cls, start, start_osmid, end_osmid, end, route,
                          instructions, id):
        """Construct an Entity from the start and end points, route, and pivots.
        """
        return RVSData(
            util.tuple_from_point(start),
            util.tuple_from_point(end),
            start_osmid,
            end_osmid,
            util.get_linestring_distance(route),
            instructions,
            id
        )
