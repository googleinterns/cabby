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

class Entity:
  """Construct a Wikigeo sample.

  `pageid` is the Wikipedia page id.
  `title` is the name of the entity.
  `text` is the Wikipedia content.
  """

  @classmethod
  def from_wiki_items(cls, wikipedia, wikipedia_ref, wikidata):
    """Construct an Entity from the Wikipedia and Wikidata entities."""
    return 
    {'pageid': wikipedia.pageid,
     'text': wikipedia.text,
    'ref_qid': wikipedia_ref.qid, 
    'ref_title': wikipedia_ref.title, 
    'ref_point': wikidata.location}


