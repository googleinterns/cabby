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

from absl import logging
import json
import os
import pandas as pd
import string
from typing import Dict, Tuple, Sequence, Text, Optional, List


from cabby.data.wikidata import query as wdq
from cabby.data.wikipedia import query as wpq
from cabby.data.wikidata import item as wdi
from cabby.data.wikipedia import item as wpi
from cabby.data.wikidata import info_item as wdqi
from cabby.data import wikigeo

from cabby.data import osm_item
from cabby.geo.map_processing import map_structure

def get_wikigeo_data(wikidata_items: Sequence[wdi.WikidataEntity]) -> List:
    '''Get data from Wikipedia based on Wikidata items" 
    Arguments:
        wikidata_items: The Wikidata items to which corresponding Wikigeo  
        items will be extracted.
    Returns:
        The Wikigeo items found (composed of Wikipedia (text,title) and 
        Wikidata (location) data).
    '''
    # Get Wikipedia titles.
    titles = [entity.wikipedia_title for entity in wikidata_items]

    # Get Wikipedia pages.
    wikipedia_pages = wpq.get_wikipedia_items(titles)
    wikipedia_items = []
    wikipedia_titles =[]
    for wikipedia_page in wikipedia_pages:
      wikipedia_page_items=[]
      for item in wikipedia_page:
        wikipedia_item = wpi.WikipediaEntity.from_api_result(item)
        wikipedia_page_items.append(wikipedia_item)
        if wikipedia_item.title not in wikipedia_titles:
          wikipedia_titles.append(wikipedia_item.title)
      wikipedia_items.append(wikipedia_page_items)

    # # Change to Geodata dataset foramt.
    geo_data = []
    for wikipedia_page_items, wikidata in zip(wikipedia_items, wikidata_items):
      for wikipedia in wikipedia_page_items:
        geo_data.append(wikigeo.WikigeoEntity.from_wiki_items(
            wikipedia, wikipedia, wikidata, "Wikipedia_page").sample)

    # # Get backlinks for Wikipedia pages.
    backlinks_pages = wpq.get_backlinks_items_from_wikipedia_titles(
        wikipedia_titles)
    backlinks_items = []
    for list_backlinks in backlinks_pages:
        backlinks_items.append(
            [wpi.WikipediaEntity.from_api_result(result) for \
                result in list_backlinks])

    # Change backlinks pages to Geodata dataset format.
    for list_backlinks, original_wikipedia_items, original_wikidata in \
        zip(backlinks_items, wikipedia_items, wikidata_items):
        for backlink in list_backlinks:
            wikigeo_sample = wikigeo.WikigeoEntity.from_wiki_items(
                backlink, original_wikipedia_items[0], original_wikidata, "Wikipedia_backlink").sample
            
            sample_text = wikigeo_sample['text']
            if any(dict['text'] == sample_text for dict in geo_data):
                continue
            geo_data.append(wikigeo_sample)
    return geo_data


def get_data_by_qid(qid: Text) -> Sequence:
  '''Get data from Wikipedia and Wikidata by region. 
  Arguments:
    qid(Text): The qid of the Wikidata to extract items from.
  Returns:
    The Wikipedia (text, title) and Wikidata (location) data found.
  '''

  # Get Wikidata items by region.
  wikidata_results = wdq.get_geofenced_wikidata_items_by_qid(qid)
  wikidata_items = [wdi.WikidataEntity.from_sparql_result(result)
            for result in wikidata_results]

  return get_wikigeo_data(wikidata_items)


def get_data_by_region(region: Text) -> Sequence:
  '''Get data from Wikipedia and Wikidata by region.
  Arguments:
    region(Text): The region to extract items from.
  Returns:
    The Wikipedia (text,title) and Wikidata (location) data found.
  '''

  # Get Wikidata items by region.
  wikidata_results = wdq.get_geofenced_wikidata_items(region)
  wikidata_items = [wdi.WikidataEntity.from_sparql_result(result)
            for result in wikidata_results]

  return get_wikigeo_data(wikidata_items)


def get_data_by_region_with_osm(region: Text, path_osm: Text = None
) -> Sequence:
  '''Get three types of samples by region: (1) samples from Wikipedia(text,title) and Wikidata(location); (2) Concatenation of Wikidata tags; (3) Concatenation of OSM tags. 
  Arguments:
    region(Text): The region to extract items from.
  Returns:
    The Wikipedia(text,title) and Wikidata(location) data found.
  '''


  # Get Wikidata items by region.
  wikidata_results = wdq.get_geofenced_wikidata_items(region)
  wikidata_items = [wdi.WikidataEntity.from_sparql_result(result)
            for result in wikidata_results]

  samples = get_wikigeo_data(wikidata_items)

  logging.info(
    'Created {} samples from Wikipedia pages and Wikidata \
    (locations only).'.format(len(samples)))

  # Add samples from Wikidata only.
  wikidata_tags = wdq.get_geofenced_info_wikidata_items(region)
  for item in wikidata_tags:
    info_item = wdqi.WikidataEntity.from_sparql_result_info(item)
    sample = wikigeo.WikigeoEntity.from_wikidata(info_item).sample
    samples.append(sample)

  logging.info(
    'Created {} samples with Wikidata additional data.'.format(len(samples)))

  # Add sample from OSM only.
  if path_osm is None:
    map = map_structure.Map(region)
    poi = map.poi
  else:
    poi = map_structure.load_poi(path_osm)
  num_cells_large_entities = 10
  # Remove large entities.
  poi = poi[poi['s2cellids'].str.len() <= num_cells_large_entities]  
  osm_entities = poi.apply(
    lambda row: osm_item.OSMEntity.from_osm(row), axis=1).tolist()
  unique_texts = []
  for osm in osm_entities:
    sample = wikigeo.WikigeoEntity.from_osm(osm).sample
    if sample['text'] not in unique_texts:
      unique_texts.append(sample['text'])
      samples.append(sample)
      
  logging.info(
    'Created {} samples with OSM additional data.'.format(len(samples)))
  return samples


def split_dataset(
    dataset: Sequence, percentage_train: float, percentage_dev: float):
  '''Splits the dataset into train-set, dev-set, test-set according to the 
  ref_qid." 
  Arguments:
    percentage_train(float in [0,1]): percentage of the train-set.
    percentage_dev(float in [0,1]): percentage of the dev-set.
  Returns:
    The train-set, dev-set and test-set splits.
  '''
  assert percentage_train >= 0 and percentage_train <= 1, \
    "percentage_train is not in range 0-1."

  assert percentage_dev >= 0 and percentage_dev <= 1, \
    "percentage_dev is not in range 0-1."

  assert percentage_dev + \
    percentage_train <= 1, "percentage_dev+percentage_train is more than 1."

  # TODO (https://github.com/googleinterns/cabby/issues/28#issue-695818890):
  # Change split by qid so that it will ensure qid isn't shared between sets

  # Sort the dataset by ref_qid.
  sorted_dataset = sorted(dataset, key=lambda item: item['ref_qid'])

  # Get size of splits.
  size_dataset = len(dataset)
  size_train = round(percentage_train*size_dataset)
  size_dev = round(percentage_dev*size_dataset)

  # Split the dataset.
  train_set = sorted_dataset[0:size_train]
  dev_set = sorted_dataset[size_train:size_train+size_dev]
  test_set = sorted_dataset[size_train+size_dev:]

  return train_set, dev_set, test_set


def write_files(path: Text, items: Sequence):
  '''Write items to disk.'''

  with open(path, 'a') as outfile:
    for item in items:
      json.dump(item, outfile, default=lambda o: o.__dict__)
      outfile.write('\n')
      outfile.flush()

