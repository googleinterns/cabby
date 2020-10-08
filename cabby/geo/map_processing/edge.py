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
class Edge:
  """Construct a Wikigeo sample.
  `u_for_edge` is one side of the segment.
  `v_for_edge` is the other side of the segment.
  `length` is the length of the segment.
  `oneway` whether it is directional.
  `highway` is always poi.
  """
  u_for_edge: int = attr.ib()
  v_for_edge: int = attr.ib()
  length: float = attr.ib()
  oneway: int = attr.ib()
  highway: Text = attr.ib()
  osmid: int = attr.ib()
  name: Text = attr.ib()

  @classmethod
  def from_projected(cls, u_for_edge, v_for_edge, length, highway, osmid, name):
    """Construct an edge entity to connect the projected point of POI.
    Arguments:
      u_for_edge: The u endside of the edge.
      v_for_edge: The v endside of the edge.
      length: length of the edge.
      highway: highway tag of the edge.
      osmid: of the edge.
      name: name of the street.
    Returns:
      An edge entity.
    """
    return Edge(
      u_for_edge,
      v_for_edge,
      length,
      False,
      highway,
      osmid,
      name
    )

  @classmethod
  def from_poi(cls, u_for_edge, v_for_edge, osmid, length):
    """Construct an edge entity to connect a POI.
    Arguments:
      u_for_edge: The u endside of the edge.
      v_for_edge: The v endside of the edge.
      osmid: of the edge.
      length: length of the edge.
    Returns:
      An edge entity.
    """
    return Edge(
      u_for_edge,
      v_for_edge,
      length,
      False,
      "poi",
      osmid,
      "poi"

    )