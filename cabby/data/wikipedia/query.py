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
import collections
import copy
import itertools
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import multiprocessing
import random
import re
import requests
import spacy
from typing import Any, Dict, Tuple, Sequence, Text, Optional

from cabby.data.wikipedia import item as wpi

# Threshold for the number of characters needed in a sentence to be extracted
# as a context from a page.
MIN_CHARACTER_COUNT = 30

# The maximum number of times a given pageid can be explored for backlink
# contexts given the set of Wikipedia titles under consideration. This is
# intended to reduce checking pages that are backlinks just because they are
# part of a Wikipedia list, such as historical places in a city.
BACKLINK_COUNT_THRESHOLD = 5

# A cap on the number of backlinks to explore per title.
MAX_BACKLINKS_PER_TITLE = 25

# Process text with spacy.
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000

def get_items_for_title(title: str) -> Sequence[wpi.WikipediaEntity]:
  '''Creates WikipediaEntity objects from the text on the requested page.

  Args:
    title: The title of the page to search for.
  Returns:
    A sequence of WikipediaEntity objects, one per sentence in the page's text. 
  '''
  url = (
    'https://en.wikipedia.org/w/api.php'
    '?action=query'
    '&prop=extracts'
    '&explaintext'
    f'&titles={title}'
    '&format=json'
    '&redirects'
  )
  try:
    json_response = requests.get(url).json()
    result = list(json_response['query']['pages'].values())[0]
  except:
    print("An exception occurred when running query: {0}".format(url))
    return []

  # Found title can be different from input title, e.g. not having underscores.
  result_title = result['title']
  pageid = result['pageid']
  text = clean_text(result['extract'])
  doc = nlp(text)

  items = []
  for span in list(doc.sents):
    for sentence in span.text.split('\n'):
      # Filter short sentences based on number of characters.
      if len(sentence) > MIN_CHARACTER_COUNT:
        items.append(wpi.WikipediaEntity(pageid, title, sentence))

  return items


def get_wikipedia_items(titles: Sequence[str]) -> Sequence[wpi.WikipediaEntity]:
  '''Query the Wikipedia API.

  Arguments:
    titles: The Wikipedia titles to run on the Wikipedia API.
  Returns:
    The Wikipedia items found.
  '''
  with multiprocessing.Pool(processes=10) as pool:
    items = pool.map(get_items_for_title, titles)
  return list(itertools.chain.from_iterable(items))


def clean_title(title: str) -> str:
  '''Parse Wikipedia title and remove unwanted chars. 
  Arguments:
    title: The Wikipedia title to be parsed.
  Returns:
    The title after parsing.
  '''
  title = title.split(',')[0]
  title = title.split('(')[0]
  return title.strip()


def clean_text(text: str) -> str:
  '''Parse Wikipedia text and remove unwanted chars. 
  Arguments:
    text(Text): The Wikipedia text to be parsed.
  Returns:
    The text after parsing.
  '''
  # Remove titles.
  text = text.split('== See also ==')[0]
  text = text.split('== External links ==')[0]
  text = text.split('== Notes ==')[0]
  text = text.split('== References ==')[0]
  text = re.sub(r'=.*=', r'.', text)
  return text


def get_backlinks_ids_from_wikipedia_title(title: str) -> Sequence[int]:
  '''Query the Wikipedia API for backlinks pageids. 
  Arguments:
    title: The Wikipedia title to obtain backlinks from.
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
    backlinks_ids = [
      y['pageid'] for k, x in json_response['query']['pages'].items() 
      for y in x['linkshere']]

  except:
    print("An exception occurred when running query: {0}".format(url))
    return []

  return backlinks_ids


def get_page_from_id(pageid: int):
  '''Gets the JSON result of looking up a page by its id.

  Args:
    pageid: The Wikipedia page id to look up.
  Returns:
    The result from the Wikipedia API for that page id.
  '''
  url = (
    'https://en.wikipedia.org/w/api.php'
      '?action=query'
      '&prop=extracts'
      '&explaintext'
      f'&pageids={pageid}'
      '&format=json'
    )

  try:
    json_response = requests.get(url).json()
    result = list(json_response['query']['pages'].values())[0]

  except:
    print("An exception occurred when running query: {0}".format(url))
    return None

  if 'may refer to:' in result['extract']:
    return None

  return result

def get_items_matching_titles(backlink_id: int, titles: Sequence[str]):
  '''Extract paragraphs containing references to any of the provided titles.
  
  Queries the Wikipedia API to obtain the page text of the backlink_id, parses
  it with Spacy, and then checks for each of the titles. By doing multiple
  titles for the same backlink_id, we only need to process the document's text
  once. Using this, we can get examples such as this from the page for Errol
  Flynn (the backlink id) but as a context for the University of Texas at Austin
  (an example title matching a sentence in Errol Flynn's page):

    'Many of these pieces were lost until 2009, when they were rediscovered in a 
    collection at the University of Texas at Austin's Dolph Briscoe Center for 
    American History.'

  Arguments:
    backlink_id: The backlink id of the Wikipedia page to search.
    titles: The Wikipedia titles to search for in the text.
  Returns:
    The backlinks Wikipedia items found.
  '''
  result = get_page_from_id(backlink_id)
  if result is None:
    return []

  result_title = result['title']
  text = clean_text(result['extract'])
  doc = nlp(text)

  items = []

  # We need clean titles for search in text, but we need to be able to map back
  # to the original titles for creating WikipediaEntity objects.
  clean_titles = [clean_title(title) for title in titles]
  clean_title_to_original = dict(zip(clean_titles, titles))

  # See which titles have at least one match and fail fast if there is little 
  # hope of finding a match for any title.
  possibly_matching_titles = []
  for title in clean_titles:
    if title in doc.text or fuzz.token_set_ratio(doc.text, title) > 90:
      possibly_matching_titles.append(title)

  if not possibly_matching_titles:
    return items

  # Extract sentence specific contexts.
  for span in list(doc.sents):
    sub_sentences = span.text.split('\n')
    for sentence in sub_sentences:
      # Filter short sentences with an insufficient character count.
      if len(sentence) < MIN_CHARACTER_COUNT:
        continue

      for title in possibly_matching_titles:
        fuzzy_score = fuzz.token_set_ratio(sentence, title)
        if title in sentence or fuzzy_score > 90:
          items.append(wpi.WikipediaEntity(
            pageid=backlink_id, 
            title=result_title, 
            text=sentence,
            linked_title=clean_title_to_original[title]
          ))

  return items

def get_backlinks_items_from_wikipedia_titles(
  titles: Sequence[str]) -> Sequence[wpi.WikipediaEntity]:
  '''Query the Wikipedia API for backlinks pages with multiple titles. 

  Arguments:
    titles: The Wikipedia titles to search through for backlinks and contexts
      matching those titles in them.
  Returns:
    A list of WikipediaEntity objects obtained from backlinks to the given
    titles.
  '''
  title_to_backlinks = {
    title: get_backlinks_ids_from_wikipedia_title(title)
    for title in titles
  }

  backlink_counts = collections.defaultdict(int)
  for backlinks in title_to_backlinks.values():
    for bl in backlinks:
      backlink_counts[bl] += 1
  
  backlinks_to_titles = collections.defaultdict(list)
  for title, backlinks in title_to_backlinks.items():
    valid_backlinks = [bl for bl in backlinks 
                       if backlink_counts[bl] < BACKLINK_COUNT_THRESHOLD]   
    random.shuffle(valid_backlinks)
    for backlink in valid_backlinks[:MAX_BACKLINKS_PER_TITLE]:
      backlinks_to_titles[backlink].append(title)

  with multiprocessing.Pool(processes=10) as pool:
    backlinks_items = pool.starmap(
        get_items_matching_titles, backlinks_to_titles.items())

  # Flatten the sequence of sequences and return the items.
  return list(itertools.chain.from_iterable(backlinks_items))
