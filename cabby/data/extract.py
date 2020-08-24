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

'''Library to support data extraction from Wikipedia and Wikidata.'''

import json
import os
from typing import Dict, Tuple, Sequence, Text
from cabby.data.wikidata import query as wdq
from cabby.data.wikipedia import query as wpq
from wikidata import item as wdi
from wikipedia import item as wpi
import wikigeo


def get_data_by_region(region: Text) -> Sequence:
    '''Get data from Wikipedia and Wikidata by region" 
    Arguments:
        region(Text): The region to extract items from.
    Returns:
        The Wikipedia (text,title) and Wikidata (location) data found.
    '''

    # Get Wikidata items by region.
    wikidata_results = wdq.get_geofenced_wikidata_items(region)
    wikidata_items = [wdi.Entity.from_sparql_result(result)
                      for result in wikidata_results]

    # Get Wikipedia titles.
    titles = [entity.wikipedia_title for entity in wikidata_items]

    # Get Wikipedia pages.
    wikipedia_pages = wpq.get_wikipedia_items(titles)
    wikipedia_items = [wpi.Entity.from_api_result(
        result) for result in wikipedia_pages]

    print("number of pages: ", len(wikipedia_items))

    # # Get Wikipedia titles.
    wikipedia_titles = [entity.title for entity in wikipedia_items]

    # # Change to Geodata dataset foramt.
    geo_data = []
    for wikipedia, wikidata in zip(wikipedia_items, wikidata_items):
        geo_data.append(wikigeo.Entity.from_wiki_items(
            wikipedia, wikipedia, wikidata).sample)

    # # Get backlinks for Wikipedia pages.
    backlinks_pages = wpq.get_backlinks_items_from_wikipedia_titles(
        wikipedia_titles)
    backlinks_items = []
    for list_backlinks in backlinks_pages:
        backlinks_items.append(
            [wpi.Entity.from_backlinks_api_result(result) for result in list_backlinks])

    print("number of backlinks: ", len(
        [y for x in backlinks_items for y in x]))

    # Change backlinks pages to Geodata dataset format.
    for list_backlinks, original_wikipedia, original_wikidata in zip(backlinks_items, wikipedia_items, wikidata_items):
        for backlink in list_backlinks:
            geo_data.append(wikigeo.Entity.from_wiki_items(
                backlink, original_wikipedia, original_wikidata).sample)

    print("total number: ", len(geo_data))
    return geo_data


