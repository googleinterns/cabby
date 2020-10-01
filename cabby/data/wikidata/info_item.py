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
from typing import Text, Dict

import attr
from shapely.geometry.point import Point

# Regular expression to pull the latitude and longitude out of a Point that is
# represented as a string. Ignores situation where multiple points are provided,
# taking just the first in that case.
_POINT_RE = re.compile(r'^Point\(([-\.0-9]+)\s([-\.0-9]+)\).*$')


@attr.s
class WikidataEntity:
  """Simplifed representation of a Wikidata entity.

  `url` is the URL of the entity.
  `title` is the name of the entity.
  `location` is a Point representing the geo-location of the entity.
  `instance` is the Wikidata instance property.    
  `qid` is the Wikidata id assigned to this entity (which can be recovered from
    the URL, but is pulled out for convenience).
  `tags` is the tags of the entity.
  `sample` is created from a concatenation of the entity's tags.

  """
  url: Text = attr.ib()
  title: Text = attr.ib()
  location: Point = attr.ib()
  instance: Text = attr.ib()
  qid: Text = attr.ib(init=False)
  tags: Dict = attr.ib()
  sample: Text = attr.ib(init=False)

  def __attrs_post_init__(self):
    # The QID is the part of the URL that comes after the last / character.
    self.qid = self.url[self.url.rindex('/')+1:]

    self.sample = create_sample_from_tags(self.tags)
  

  @classmethod
  def from_sparql_result_info(cls, result):
    """Construct an Entity from the results of a SPARQL query."""
    point_match = _POINT_RE.match(result['point']['value'])
    return WikidataEntity(
        result['place']['value'],
        result['placeLabel']['value'],
        Point(float(point_match.group(1)), float(point_match.group(2))),
        result['instance']['value'],
        result
        
    )
def create_sample_from_tags(item: Dict) -> Text:
  '''Get a Wikidata item and return a sample based on tags." 
  Arguments:
      item: The Wikidata item to create a sample from.
  Returns:
      A sample.
  '''
  unwanted_tags = ['place', 'point']
  sample = ""
  connect = " and "
  for k, v in item.items():
    if k not in unwanted_tags:
      sample+=connect + v['value']
  sample = sample[len(connect):] + "." # Remove 'and' in the beginning.
  
  return sample

