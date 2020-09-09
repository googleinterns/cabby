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
    """
    pageid: int = attr.ib()
    text: Text = attr.ib()
    title: Text = attr.ib()
    ref_qid: Text = attr.ib()
    ref_pageid: Text = attr.ib()
    ref_title: Text = attr.ib()
    ref_point: Point = attr.ib()
    ref_instance: Point = attr.ib()
    sample: dict = attr.ib(init=False)

    def __attrs_post_init__(self):
        # The QID is the part of the URL that comes after the last / character.
        self.sample = {
            'pageid': self.pageid, 'text': self.text, 'title': 
            self.title, 'ref_qid': self.ref_qid, 'ref_pageid': self.ref_pageid, 'ref_title': self.ref_title, 'ref_point': text_from_point(self.ref_point), 'ref_instance': self.ref_instance
        }

    @classmethod
    def from_wiki_items(cls, wikipedia, wikipedia_ref, wikidata_ref):
        """Construct an Entity from the Wikipedia and Wikidata entities."""
        return WikigeoEntity(
            wikipedia.pageid,
            wikipedia.text,
            wikipedia.title,
            wikidata_ref.qid,
            wikipedia_ref.pageid,
            wikipedia_ref.title,
            wikidata_ref.location,
            wikidata_ref.instance)


def text_from_point(point: Point) -> Text:
    '''Convert a Point into a tuple, with latitude as first element, and longitude as second.
    Arguments:
      point(Point): A lat-lng point.
    Returns:
      A lat-lng text.
    '''

    return "({0}, {1})".format(point.y, point.x)
