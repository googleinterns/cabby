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

'''Library to support Wikipedia queries.'''


from typing import Dict, Tuple, Sequence, Text
import requests
import spacy


def get_wikipedia_items(titles: Sequence[Text]) -> Dict:
    '''Query the Wikipedia API. 
    Arguments:
        titles(Text): The wikipedia titles to run on the Wikipedia API.
    Returns:
        The Wikipedia items found.
    '''

    string_titles = '|'.join(titles)
    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&prop=extracts'
        '&exintro'
        '&exsentences=3'
        '&explaintext'
        f'&titles={string_titles}'
        '&format=json'
    )
    json_response = requests.get(url).json()
    entities = json_response['query']['pages']
    return entities


def get_wikipedia_items_title_in_text(backlinks_ids: Sequence[Text], orig_title: Text) -> Sequence[Dict]:
    '''Query the Wikipedia API. 
    Arguments:
        backlinks_titles(Sequence[Text]): The wikipedia backlinks titles to run on the Wikipedia API.
        orig_title(Text): The wikipedia title that would be searched in the beacklinks text.
    Returns:
        The backlinks Wikipedia items found.
    '''

    # Process text with spacy.
    nlp = spacy.load("en_core_web_sm")

    # Title as it will appear in th text.
    orig_title = orig_title.replace('_', ' ')

    list_entities = []

    for backlink_id in backlinks_ids:

        url = (
            'https://en.wikipedia.org/w/api.php'
            '?action=query'
            '&prop=extracts'
            '&explaintext'
            f'&pageids={backlink_id}'
            '&format=json'
        )
        json_response = requests.get(url).json()
        entities = json_response['query']['pages']
        entity = next(iter(entities.values()))

        doc = nlp(entity['extract'])
        sentences = list(doc.sents)

        entity['sentences'] = [
            x.text for x in sentences if orig_title in x.text]
        if len(entity['sentences']) == 0:
            continue

        list_entities.append(entity)

    return list_entities


def get_baklinks_ids_from_wikipedia_title(title: Text) -> Sequence[Text]:
    '''Query the Wikipedia API for backlinks titles. 
    Arguments:
        title(Text): The wikipedia title for which the backlinks will be connected to.
    Returns:
        The backlinks titles.
    '''

    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&list=backlinks'
        f'&bltitle={title}'
        '&format=json'
    )
    json_response = requests.get(url).json()

    backlinks_pageid = [x['pageid']
                        for x in json_response['query']['backlinks']]
    return backlinks_pageid


def get_baklinks_items_from_wikipedia_title(title: Text) -> Sequence[Dict]:
    '''Query the Wikipedia API for backlinks pages. 
    Arguments:
        title(Text): The wikipedia title for which the backlinks will be connected to.
    Returns:
        The backlinks pages.
    '''

    # Get the backlinks titles.
    backlinks_pageid = get_baklinks_ids_from_wikipedia_title(title)

    # Get the backlinks pages
    backlinks_pages = get_wikipedia_items_title_in_text(
        backlinks_pageid, title)

    return backlinks_pages
