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

import re
from typing import Text, Dict

import attr
from geopandas import GeoDataFrame
from shapely.geometry.point import Point

from cabby.geo import util


VERSION = 0.3


@attr.s()
class RVSData:
    """Construct a RVSdata sample.
    `start_point` is the beginning location.
    `end_point` is the goal location.
    `distance` path distance between start and end point.
    `instructions` the instructions describing how to get to the end point.
    `id` is the id of the entity.
    `version` version of the RVS samples.
    """
    start_point: tuple = attr.ib()
    end_point: tuple = attr.ib()
    start_osmid: int = attr.ib()
    end_osmid: int = attr.ib()
    distance: int = attr.ib()
    instructions: Text = attr.ib()
    id: int = attr.ib()
    version: float = attr.ib()


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
            id,
            VERSION
        )
