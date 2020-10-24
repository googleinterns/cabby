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
'''Basic classes and functions for Wikipedia items.'''

import re

import attr


@attr.s
class WikipediaEntity:
  """Simplifed representation of a Wikipedia entity.

  `pageid` is the Wikipedia page id.
  `title` is the name of the entity.
  `text` is the Wikipedia content.
  `linked_title` is an optional title of another Wikipedia page that is
    associated with this one. This is used when the text contains a reference
    to another page.
  """
  pageid: int = attr.ib()
  title: str = attr.ib()
  text: str = attr.ib()
  linked_title: str = attr.ib(default='')

  def __attrs_post_init__(self):
    # Remove unwanted chars from text.
    self.text = self.text.replace("\n", " ").replace("=", "")

  @classmethod
  def from_api_result(cls, result):
    """Construct an Entity from the results of the Wikimedia API query."""
    return WikipediaEntity(result['pageid'], result['title'], result['extract'])