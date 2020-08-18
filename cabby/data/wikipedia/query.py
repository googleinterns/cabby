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


from typing import Dict, Tuple, Sequence, Text, Optional
import requests
import spacy
import multiprocessing


def get_wikipedia_item(title: Text) -> Dict:

    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&prop=extracts'
        '&exintro'
        '&exsentences=3'
        '&explaintext'
        f'&titles={title}'
        '&format=json'
    )
    json_response = requests.get(url).json()
    return json_response['query']['pages']


def get_wikipedia_items(titles: Sequence[Text]) -> Sequence:
    '''Query the Wikipedia API. 
    Arguments:
        titles(Text): The Wikipedia titles to run on the Wikipedia API.
    Returns:
        The Wikipedia items found.
    '''

    with multiprocessing.Pool(processes=4) as pool:
        entities = pool.map(get_wikipedia_item,
                            titles)

    return entities


def get_wikipedia_items_title_in_text(backlink_id: Text, orig_title: Text) -> Optional[Sequence[Dict]]:
    '''Query the Wikipedia API. 
    Arguments:
        backlinks_titles(Sequence[Text]): The Wikipedia backlinks titles to
        run on the Wikipedia API.
        orig_title(Text): The Wikipedia title that would be searched in
        the beacklinks text.
    Returns:
        The backlinks Wikipedia items found.
    '''

    # Process text with spacy.
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000

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

    if orig_title not in doc.text:
        return

    sentences = list(doc.sents)

    entity['sentences'] = [x.text for x in sentences if orig_title in x.text]

    if len(entity['sentences']) > 0:

        return entity

    return


def get_baklinks_ids_from_wikipedia_title(title: Text) -> Sequence[Text]:
    '''Query the Wikipedia API for backlinks titles. 
    Arguments:
        title(Text): The Wikipedia title for which the backlinks
        will be connected to.
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

    backlinks_ids = [x['pageid'] for x in json_response['query']['backlinks']]
    return backlinks_ids


def get_backlinks_items_from_wikipedia_title(title: Text) -> Sequence[Sequence]:
    '''Query the Wikipedia API for backlinks pages. 
    Arguments:
        title(Text): The Wikipedia title for which the backlinks
        will be connected to.
    Returns:
        The backlinks pages.
    '''
    # Get the backlinks titles.
    backlinks_pageids = get_baklinks_ids_from_wikipedia_title(title)

    # Title as it will appear in th text.
    title = title.replace('_', ' ')

    # Get the backlinks pages.
    backlinks_pages = []

    for id in backlinks_pageids:
        backlinks_page = get_wikipedia_items_title_in_text(id, title)
        if backlinks_page is not None:
            backlinks_pages.append(backlinks_page)

    return backlinks_pages


def get_backlinks_items_from_wikipedia_titles(titles: Sequence[Text]) -> Sequence[Dict]:
    '''Query the Wikipedia API for backlinks pages multiple titles. 
    Arguments:
        titles(Sequence): The Wikipedia titles for which the
        backlinks will be connected to.
    Returns:
        The backlinks pages.
    '''

    with multiprocessing.Pool(processes=4) as pool:
        backlinks_pages = pool.map(get_backlinks_items_from_wikipedia_title,
                                   titles)

    return backlinks_pages[0]
