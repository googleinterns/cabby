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
from typing import Dict, List, Sequence

from cabby.data.wikidata import query as wdq
from cabby.data.wikipedia import query as wpq
from cabby.data.wikidata import item as wdi
from cabby.data.wikipedia import item as wpi
from cabby.data.wikidata import info_item as wdqi
from cabby.data import osm_item
from cabby.data import wikigeo
from cabby.geo.map_processing import map_structure
from cabby.geo.regions import Region

def get_wikigeo_data(
    wikidata_items: Sequence[wdi.WikidataEntity]
) -> List[wikigeo.WikigeoEntity]:
    '''Get data from Wikipedia based on Wikidata items" 
    Arguments:
        wikidata_items: The Wikidata items to which corresponding Wikigeo  
        items will be extracted.
    Returns:
        The Wikigeo items found (composed of Wikipedia (text,title) and 
        Wikidata (location) data).
    '''
    # Get Wikipedia titles.
    titles_to_wikidata = {item.wikipedia_title: item for item in wikidata_items}

    # Get Wikipedia pages.
    wikipedia_items = wpq.get_wikipedia_items(list(titles_to_wikidata.keys()))
    titles_to_wikipedia = {item.title: item for item in wikipedia_items}

    # Change to Geodata dataset format.
    geo_data = []
    for wp_item in wikipedia_items:
      geo_data.append(wikigeo.WikigeoEntity.from_wiki_items(
        wp_item, wp_item, titles_to_wikidata[wp_item.title], 'Wikipedia_page'))

    # Get backlinks for Wikipedia pages.
    logging.info("Creating WikipediaEntities from backlinks.")
    backlinks_items = wpq.get_backlinks_items_from_wikipedia_titles(
      list(titles_to_wikidata.keys()))

    # Change backlinks pages to Geodata dataset format.
    logging.info("Converting backlink pages to geodata")

    for item in backlinks_items:
      geo_data.append(wikigeo.WikigeoEntity.from_wiki_items(
        item, 
        titles_to_wikipedia[item.linked_title], 
        titles_to_wikidata[item.linked_title], 
        "Wikipedia_backlink"))

    # Remove duplicates.
    logging.info("Removing duplicate items.")
    uniq_wgitems = {}
    for wgitem in geo_data:
      key = wgitem.ref_qid + wgitem.text
      uniq_wgitems[key] = wgitem

    logging.info("Done with all extraction.")

    return list(uniq_wgitems.values())


def get_data_by_qid(qid: str) -> Sequence[wikigeo.WikigeoEntity]:
  '''Get data from Wikipedia and Wikidata by region. 
  Arguments:
    qid(str): The qid of the Wikidata to extract items from.
  Returns:
    The Wikipedia (text, title) and Wikidata (location) data found.
  '''

  # Get Wikidata items by qid.
  wikidata_results = wdq.get_place_location_points_from_qid(qid)
  wikidata_items = [wdi.WikidataEntity.from_sparql_result(result)
                    for result in wikidata_results]

  return get_wikigeo_data(wikidata_items)


def get_data_by_region(region: Region) -> List[wikigeo.WikigeoEntity]:
  '''Get data from Wikipedia and Wikidata by region.
  Arguments:
    region(Region): The region to extract items from.
  Returns:
    The Wikipedia (text,title) and Wikidata (location) data found.
  '''

  # Get Wikidata items by region.
  wikidata_results = wdq.get_geofenced_wikidata_items(region)
  wikidata_items = [wdi.WikidataEntity.from_sparql_result(result)
                    for result in wikidata_results]
  return get_wikigeo_data(wikidata_items)


def get_data_by_region_with_osm(
    region: Region, path_osm: str = None) -> List[wikigeo.WikigeoEntity]:
  '''Get three types of samples by region: 
      (1) samples from Wikipedia(text, title) and Wikidata(location)
      (2) Concatenation of Wikidata tags
      (3) Concatenation of OSM tags. 

  Arguments:
    region(Region): The region to extract items from.
  Returns:
    The Wikipedia(text,title) and Wikidata(location) data found.
  '''

  # Get Wikidata items by region.
  samples = get_data_by_region(region)

  logging.info(
    f'Created {len(samples)} samples from Wikipedia pages and Wikidata (locations only).')

  # Add samples from Wikidata only.
  wikidata_tags = wdq.get_geofenced_info_wikidata_items(region)
  for item in wikidata_tags:
    info_item = wdqi.WikidataEntity.from_sparql_result_info(item)
    sample = wikigeo.WikigeoEntity.from_wikidata(info_item)
    samples.append(sample)

  logging.info(
    f'Created {len(samples)} samples with Wikidata additional data.')

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
    sample = wikigeo.WikigeoEntity.from_osm(osm)
    if sample.text not in unique_texts:
      unique_texts.append(sample.text)
      samples.append(sample)
      
  logging.info(
    f'Created {len(samples)} samples with OSM additional data.')
  return samples


def split_dataset(
    dataset: Sequence[wikigeo.WikigeoEntity],
    percentage_train: float,
    percentage_dev: float
) -> Dict[str, Sequence[wikigeo.WikigeoEntity]]:
  '''Splits the dataset into train-set, dev-set, test-set according to the 
  ref_title." 
  Arguments:
    percentage_train(float in [0,1]): percentage of the train-set.
    percentage_dev(float in [0,1]): percentage of the dev-set.
  Returns:
    The train-set, dev-set and test-set splits.
  '''
  assert percentage_train >= 0 and percentage_train <= 1, (
    "percentage_train is not in range 0-1.")

  assert percentage_dev >= 0 and percentage_dev <= 1, (
    "percentage_dev is not in range 0-1.")

  assert percentage_dev + percentage_train <= 1, (
    "percentage_dev+percentage_train is more than 1.")

  # TODO (https://github.com/googleinterns/cabby/issues/28#issue-695818890):
  # Change split by qid so that it will ensure qid isn't shared between sets

  # Sort the dataset by ref_title.
  sorted_dataset = sorted(dataset, key=lambda item: item.ref_title)

  # Get size of splits.
  size_dataset = len(dataset)
  size_train = round(percentage_train*size_dataset)
  size_dev = round(percentage_dev*size_dataset)

  # Split the dataset.
  train_set = sorted_dataset[0:size_train]
  dev_set = sorted_dataset[size_train:size_train+size_dev]
  test_set = sorted_dataset[size_train+size_dev:]

  return {
    'train': train_set, 
    'dev': dev_set,
    'test': test_set
  }


def write_files(path: str, items: Sequence):
  '''Write items to disk.'''

  with open(path, 'a') as outfile:
    for item in items:
      json.dump(item, outfile, default=lambda o: o.__dict__)
      outfile.write('\n')
      outfile.flush()
