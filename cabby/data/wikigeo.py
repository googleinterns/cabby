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
from typing import Text
from shapely.geometry.point import Point

import attr

from cabby.data.wikidata import item as wdi
from cabby.data.wikipedia import item as wpi

VERSION = 0.17

@attr.s
class WikigeoEntity:
  """Construct a Wikigeo sample.

  `pageid` is the Wikipedia page id.
  `text` is the Wikipedia content.
  `title` is the Wikipedia title.
  `ref_qid` is the Wikidata id assigned to this entity.
  `ref_pageid` is the Wikipedia page id assigned to this entity.
  `ref_title` is the refrenced Wikipedia page title.    
  `ref_point` is the refrenced Wikidata location.    
  `ref_instance` is the refrenced Wikidata instance property.    
  `sample_type` there are 4 types: (1) Wikipedia_page; (2) Wikipedia_backlink; (3) Wikidata; (4) OSM.    
  `version` the version WikiGeo sample.    

  """
  pageid: int = attr.ib()
  text: Text = attr.ib()
  title: Text = attr.ib()
  ref_qid: Text = attr.ib()
  ref_pageid: int = attr.ib()
  ref_title: Text = attr.ib()
  ref_point: Point = attr.ib()
  ref_instance: Text = attr.ib()
  sample_type: Text = attr.ib()
  version: int = attr.ib(init=False)

  def __attrs_post_init__(self):
    # The QID is the part of the URL that comes after the last / character.
    self.ref_point = text_from_point(self.ref_point)
    self.version = VERSION

  @classmethod
  def from_wiki_items(
    cls, 
    wikipedia: wpi.WikipediaEntity, 
    wikipedia_ref: wpi.WikipediaEntity, 
    wikidata_ref: wdi.WikidataEntity, 
    wikipedia_type: str):
    """Construct an Entity from the Wikipedia and Wikidata entities."""
    return WikigeoEntity(
      wikipedia.pageid,
      wikipedia.text,
      wikipedia.title,
      wikidata_ref.qid,
      wikipedia_ref.pageid,
      str(wikipedia_ref.title),
      wikidata_ref.location,
      wikidata_ref.instance,
      wikipedia_type)
  
  @classmethod
  def from_wikidata(cls, result):
    """Construct an Entity from the Wikidata entity."""
    return WikigeoEntity(
      -1,
      result.text,
      result.title,
      result.qid,
      -1,
      str(result.title),
      result.location,
      result.instance,
      "Wikidata")

  @classmethod
  def from_osm(cls, result):
    """Construct an Entity from an osm entity."""
    return WikigeoEntity(
      result.osmid,
      result.text,
      result.name,
      str(result.qid),
      result.osmid,
      str(result.name),
      result.point,
      "",
      "OSM")

def text_from_point(point: Point) -> Text:
  '''Convert a Point into a tuple, with latitude as first element, and longitude as second.
  Arguments:
    point(Point): A lat-lng point.
  Returns:
    A lat-lng text.
  '''

  return "({0}, {1})".format(point.y, point.x)
