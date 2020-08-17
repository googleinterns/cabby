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
'''Basic functions for Wikidata items.'''

import re
from typing import Text

import attr
from shapely.geometry.point import Point

# Ignores situation where multiple points are provided, taking just the first.
_POINT_RE = re.compile(r'^Point\(([-\.0-9]+)\s([-\.0-9]+)\).*$')


@attr.s
class Entity:
  url: Text = attr.ib()
  label: Text = attr.ib()
  location: Point = attr.ib()
  qid: Text = attr.ib(init=False)

  def __attrs_post_init__(self):
    # The QID is the part of the URL that comes after the last / character.
    self.qid = self.url[self.url.rindex('/')+1:]

  @classmethod
  def from_sparql_result(cls, result):
    point_match = _POINT_RE.match(result['point']['value'])
    return Entity(
        result['place']['value'],
        result['placeLabel']['value'],
        Point(float(point_match.group(1)), float(point_match.group(2)))
    )
