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
'''Basic classes and functions for Wikidata items.'''

import re
from typing import Text

import attr
from shapely.geometry.point import Point

# Regular expression to pull the latitude and longitude out of a Point that is
# represented as a string. Ignores situation where multiple points are provided,
# taking just the first in that case.
_POINT_RE = re.compile(r'^Point\(([-\.0-9]+)\s([-\.0-9]+)\).*$')


@attr.s
class Entity:
  """Simplifed representation of a Wikidata entity.

  `url` is the URL of the entity.
  `label` is the name of the entity.
  `location` is a Point representing the geo-location of the entity.
  `qid` is the Wikidata id assigned to this entity (which can be recovered from
    the URL, but is pulled out for convenience).
  """
  url: Text = attr.ib()
  label: Text = attr.ib()
  location: Point = attr.ib()
  qid: Text = attr.ib(init=False)

  def __attrs_post_init__(self):
    # The QID is the part of the URL that comes after the last / character.
    self.qid = self.url[self.url.rindex('/')+1:]

  @classmethod
  def from_sparql_result(cls, result):
    """Construct an Entity from the results of a SPARQL query."""
    point_match = _POINT_RE.match(result['point']['value'])
    return Entity(
        result['place']['value'],
        result['placeLabel']['value'],
        Point(float(point_match.group(1)), float(point_match.group(2)))
    )
