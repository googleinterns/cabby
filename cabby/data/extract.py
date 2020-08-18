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

from typing import Dict, Tuple, Sequence, Text
from cabby.data.wikidata import query as wdq
from cabby.data.wikipedia import query as wpq


def get_data_by_region(region: Text) -> Sequence[Dict]:
    '''Get data from Wikipedia and Wikidata by region" 
    Arguments:
        region(Text): The region to extract items from.
    Returns:
        The Wikipedia (text,title) and Wikidata (location) data found.
    '''

    # Get Wikidata items by region.
    wikidata_items = wdq.get_geofenced_wikidata_items(region)[
        'results']['bindings']

    points = [x['point'] for x in wikidata_items]

    # Get Wikipedia titles.
    titles = [x['wikipediaUrl']['value'].split(
        "/")[-1] for x in wikidata_items]

    # Get Wikipedia pages.
    wikipedia_pages = wpq.get_wikipedia_items(titles)

    # Change to Geodata dataset foramt.
    geo_data = []
    for x in wikipedia_pages:
        for (k, v), p in zip(x.items(), points):
            geo_data.append(
                {'extract': v['extract'], 'pageid': v['pageid'], 'title': v['title'], 'point': p['value']})

    # Get backlinks for Wikipedia pages.
    back_links_pages = wpq.get_backlinks_items_from_wikipedia_titles(
        titles)

    # Change backlinks pages to Geodata dataset format.
    geo_data_backlinks = []
    for x, p, w in zip(back_links_pages, points, geo_data):
        geo_data_backlinks.append(
            {'extract': x['extract'], 'pageid': w['pageid'], 'title': w['title'], 'point': p['value']})

    # Get add the backlinks to Wikipedia pages and get unique list.
    geo_data = geo_data + geo_data_backlinks
    geo_data = list({v['pageid']: v for v in geo_data}.values())

    return geo_data
