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
import copy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

# Process text with spacy.
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000


def get_wikipedia_item(title: Text) -> Optional[Sequence]:

    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&prop=extracts'
        '&exintro'
        '&exsentences=3'
        '&explaintext'
        f'&titles={title}'
        '&format=json'
        '&redirects'
    )
    try:
        json_response = requests.get(url).json()
    except:
        print("An exception occurred when runing query: {0}".format(url))
        return []

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


def clean_title(title: Text) -> Text:
    '''Parse Wikipedia title and remove unwanted chars. 
    Arguments:
      title(Text): The Wikipedia title to be parsed.
    Returns:
      The title after parsing.
    '''
    title = title.split(',')[0]
    title = title.split('(')[0]
    return title

def clean_text(text: Text) -> Text:
    '''Parse Wikipedia text and remove unwanted chars. 
    Arguments:
      text(Text): The Wikipedia text to be parsed.
    Returns:
      The text after parsing.
    '''
    # Remove titles.
    clean_text = text.split('== See also ==')[0]
    clean_text = clean_text.split('== Notes ==')[0]
    clean_text = re.sub(r'=.*=', r'.', clean_text)
    return clean_text


def get_wikipedia_items_title_in_text(backlink_id: int, orig_title: Text) -> Sequence:
    '''Query the Wikipedia API. 
    Arguments:
      backlinks_titles(Sequence[Text]): The Wikipedia backlinks titles to
      run on the Wikipedia API.
      orig_title(Text): The Wikipedia title that would be searched in
      the backlinks text.
    Returns:
      The backlinks Wikipedia items found.
    '''

    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&prop=extracts'
        '&explaintext'
        f'&pageids={backlink_id}'
        '&format=json'
    )

    orig_title = clean_title(orig_title)

    try:
        json_response = requests.get(url).json()
        entity = list(json_response['query']['pages'].values())[0]

    except:
        print("An exception occurred when runing query: {0}".format(url))
        return []

    if 'may refer to:' in entity['extract']:
        return []

    text = clean_text(entity['extract'])

    doc = nlp(text)

    entities = []

    fuzzy_score = fuzz.token_set_ratio(doc.text, orig_title)

    if orig_title not in doc.text and fuzzy_score < 90:
        return entities

    for span in list(doc.sents):
        sub_sentences = span.text.split('\n')
        for sentence in sub_sentences:
            fuzzy_score = fuzz.token_set_ratio(sentence, orig_title)
            if orig_title not in sentence and fuzzy_score < 90:
                continue

            sub_entity = copy.deepcopy(entity)
            sub_entity['extract'] = sentence
            entities.append(sub_entity)

    return entities


def get_backlinks_ids_from_wikipedia_title(title: Text) -> Sequence[int]:
    '''Query the Wikipedia API for backlinks pageids. 
    Arguments:
      title(Text): The Wikipedia title for which the backlinks
      will be connected to.
    Returns:
      The backlinks pageids.
    '''

    url = (
        'https://en.wikipedia.org/w/api.php'
        '?action=query'
        '&prop=linkshere'
        f'&titles={title}'
        '&format=json'
        '&lhlimit=500'
        '&lhnamespace=0'
    )

    try:
        json_response = requests.get(url).json()

        backlinks_ids = [y['pageid'] for k, x in json_response['query']
                         ['pages'].items() for y in x['linkshere']]

    except:
        print("An exception occurred when runing query: {0}".format(url))
        return []

    return backlinks_ids


def get_backlinks_items_from_wikipedia_title(title: Text) -> Sequence:
    '''Query the Wikipedia API for backlinks pages. 
    Arguments:
      title(Text): The Wikipedia title for which the backlinks
      will be connected to.
    Returns:
      The backlinks pages.
    '''
    # Get the backlinks titles.
    backlinks_pageids = get_backlinks_ids_from_wikipedia_title(title)

    # Get the backlinks pages.
    backlinks_pages = []

    for id in backlinks_pageids:
        backlinks_pages += get_wikipedia_items_title_in_text(id, title)
    return backlinks_pages


def get_backlinks_items_from_wikipedia_titles(titles: Sequence[Text]) -> Sequence[Sequence]:
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

    return backlinks_pages
