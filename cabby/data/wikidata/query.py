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

'''Library to support Wikidata geographic queries from https://query.wikidata.org/sparql.'''

from typing import Dict, Tuple, Sequence, Text, Any, List
import sys

import pandas as pd
from shapely.geometry.point import Point
from SPARQLWrapper import SPARQLWrapper, JSON

from cabby.geo import util
from cabby.geo.regions import Region

def create_info_query_from_region(region: Region) -> str:
  return create_info_query(region.corner_sw, region.corner_ne)

def create_info_query(corner_west: Point, corner_east: Point) -> str:
  return """
    SELECT ?place ?placeLabel 
          ?placeDescription ?architecturalStyleLabel ?subsidiaryLabel 
          ?useLabel ?hasPartLabel
          ( GROUP_CONCAT ( DISTINCT ?altLabel; separator="; " ) AS ?altLabelList )
          ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
          (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
          WHERE 
          {
          {
            ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            OPTIONAL {?place wdt:P527 ?hasPart}.
            OPTIONAL {?place wdt:P366 ?use}.
            OPTIONAL {?place wdt:P355 ?subsidiary}.
            OPTIONAL {?place wdt:P149 ?architecturalStyle}.
            OPTIONAL { ?place skos:altLabel ?altLabel . FILTER (lang(?altLabel) = "en") }
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
    """ + f"""
            bd:serviceParam wikibase:cornerWest "Point({corner_west.x},{corner_west.y})"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point({corner_east.x},{corner_east.y})"^^geo:wktLiteral .
    """ + """
            }
          }
          FILTER (?instance  not in
          (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
          wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
          wd:Q744913, wd:Q186117, wd:Q3298291) )
          }
          GROUP BY ?place ?placeLabel ?wikipediaUrl ?placeDescription ?architecturalStyleLabel ?subsidiaryLabel ?useLabel ?hasPartLabel
    """

def create_query_from_region(region: Region) -> str:
  return create_query(region.corner_sw, region.corner_ne)

def create_query(corner_west: Point, corner_east: Point) -> str:
  return """
    SELECT ?place ?placeLabel ?wikipediaUrl
           ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
          (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
          WHERE 
          {
          {
            ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
      """ + f"""
            bd:serviceParam wikibase:cornerWest "Point({corner_west.x},{corner_west.y})"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point({corner_east.x},{corner_east.y})"^^geo:wktLiteral .
      """ + """
            }
           FILTER (?instance  not in
          (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
          wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
          wd:Q744913, wd:Q186117, wd:Q3298291) )
          }
            UNION
                      {
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
      """ + f"""
            bd:serviceParam wikibase:cornerWest "Point({corner_west.x},{corner_west.y})"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point({corner_east.x},{corner_east.y})"^^geo:wktLiteral .
      """ + """
            }
          }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
          }
          GROUP BY ?place ?placeLabel ?wikipediaUrl
  """

def create_relation_query_from_region(region: Region) -> str:
  return create_relation_query(region.corner_sw, region.corner_ne)

def create_relation_query(corner_west: Point, corner_east: Point) -> str:
  return """
    SELECT ?place ?placeLabel ?p ?propLabel ?instance ?instanceLabel
          WHERE 
          {
          {
            ?place ?p ?instance.
            FILTER (?p IN (wdt:P31,
                           wdt:P5353,
                           wdt:P2012,
                           wdt:P361,
                           wdt:P149,
                           wdt:P84,
                           wdt:P138,
                           wdt:P112,
                           wdt:P1435,
                           wdt:P1640,
                           wdt:P463,
                           wdt:P355,
                           wdt:P527,
                           wdt:P140) )
            %s
            # SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } 
              ?prop wikibase:directClaim ?p .
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
    """ + f"""
            bd:serviceParam wikibase:cornerWest "Point({corner_west.x},{corner_west.y})"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point({corner_east.x},{corner_east.y})"^^geo:wktLiteral .
    """ + """
            }
          }
          FILTER (?instance  not in
          (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
          wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
          wd:Q744913, wd:Q186117, wd:Q3298291) )
          }
    """

_BY_QID_QUERY_LOCATION_ONLY = """SELECT ?place
          (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
          WHERE 
          {
            VALUES ?place {wd:%s}
            ?place wdt:P625 ?location .
            ?wikipediaUrl schema:about ?place.
          }
          GROUP BY ?place
        """

_BY_QID_QUERY = """SELECT ?place ?placeLabel ?wikipediaUrl
           ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
          (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
          WHERE 
          {
          {
            VALUES ?place {wd:%s}
            ?place wdt:P31 ?instance.
            ?place wdt:P625 ?location .
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
          }
          }
          GROUP BY ?place ?placeLabel ?wikipediaUrl
        """

def create_location_query_from_region(region: Region) -> str:
  return create_location_query(region.corner_sw, region.corner_ne)

def create_location_query(corner_west: Point, corner_east: Point) -> str:
  return """
    SELECT ?place ?location
          WHERE 
          {
          {
            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
    """ + f"""
            bd:serviceParam wikibase:cornerWest "Point({corner_west.x},{corner_west.y})"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point({corner_east.x},{corner_east.y})"^^geo:wktLiteral .
    """ + """
            }
            %s
          }
          }
    """

def get_geofenced_info_wikidata_items(region: Region) -> List[Dict[Text, Any]]:

  '''Get Wikidata items with extensive intformation for a specific area. 
  Arguments:
    region(Region): The area to query the Wikidata items.
  Returns:
    The Wikidata items with extensive information found in the area.
  '''
  results = query_api(create_info_query_from_region(region))

  # Filter by map region.
  filtered_results = []
  for result in results:
    point_str = result['point']['value']
    point = util.point_str_to_shapely_point(point_str) 
    if region.polygon.contains(point):
      filtered_results.append(result)
  
  return filtered_results


def get_geofenced_wikidata_items(region: Region) -> Sequence[Dict[Text, Any]]:

  '''Get Wikidata items for a specific area. 
  Arguments:
    region(Region): The area to query the Wikidata items.
  Returns:
    The Wikidata items found in the area.
  '''
  results = query_api(create_query_from_region(region))

  # Filter by map region.
  filtered_results = []
  for result in results:
    point_str = result['point']['value']
    point = util.point_str_to_shapely_point(point_str) 
    if region.polygon.contains(point):
      filtered_results.append(result)
  return filtered_results

def get_filter_string(place_filter: Sequence[Text],
                      place_param: Text = "place"):
  """Get an appropriate FILTER sparql command for the input sequence.
  Arguments:
    place_filter: list of wd IDs as strings.
    place_param: the name of the parameter to filter on.
  Returns:
    filter_string: a string like "FILTER (?place IN ...)". Returns empty string
                   if the input list is empty.
  """
  if len(place_filter) == 0:
    return ""
  filter_string = "FILTER (?%s IN (%s))" % (
    place_param,
    ",".join(["wd:%s" % qid for qid in place_filter]))
  return filter_string

def get_locations_by_qid(region: Region,
                         place_filter: Sequence[Text] = []) -> Dict[Text, Any]:
  """Get a map from QID to coordinate location in a particular region.
  Arguments:
    region(Text): region to query.
    place_filter: a list of QIDs (e.g. ["Q123", "Q987"]) to filter the places.
                  If left empty, no place filtering will happen.
  Returns:
    locations: map from QID (string) to shapely Point
  """
  query_result = query_api(create_location_query_from_region(region) 
                           % get_filter_string(place_filter))
  locations = {}
  for result in query_result:
    qid = result['place']['value'].rsplit("/", 1)[1]
    point = util.point_str_to_shapely_point(result['location']['value'])
    locations[qid] = point
  return locations

def get_geofenced_wikidata_relations(region: Region,
                                     place_filter: Sequence[Text] = [],
                                     extract_qids = False) -> pd.DataFrame:
  '''Get Wikidata relations for a specific area.
  Arguments:
    region(Region): The area to query the Wikidata items.
    place_filter: a list of QIDs (e.g. ["Q123", "Q987"]) to filter the places.
                  If left empty, no place filtering will happen.
    extract_qids: If true, the columns place, p, and instance will hold just the
                  QIDs/PIDs found in the last part of the wikidata URI.
  Returns:
    The Wikidata items, and certain relations to other Wikidata items. Columns:
      place: wikidata item corresponding to place within the region
      p: wikidata property extracted from the place
      instance: value of the property p
      instanceLabel: human-readable version of instance
      placeLabel: human-readable version of place
      propLabel: human-readable version of p
  '''
  query_result = query_api(create_relation_query_from_region(region)
                           % get_filter_string(place_filter))
  result_df = pd.DataFrame([{k: v['value'] for k, v in x.items()} for x in query_result])
  if extract_qids:
    extract_qid = lambda s: s.apply(lambda x: x.rsplit("/", 1)[1])
    extract_cols = ["place", "p", "instance"]
    result_df[extract_cols] = result_df[extract_cols].apply(extract_qid)
  return result_df

def get_place_location_points_from_qid(
  qid: Text, location_only: bool = False) -> Sequence[Dict[Text, Any]]:
  '''Get lat/long point for a particular QID.
  Arguments:
    qid(Text): The qid to return point of.
    location_only: if True, the return list will only include two dicts: one for QID ('place')
      and one for the string version of a Point ('point'). Note that if False, this may return
      null results for certain places that have non-English place/instance labels.
  Returns:
    list of SPARQLWrapper return dicts giving wikidata fields and values.
  '''
  if location_only:
    query = _BY_QID_QUERY_LOCATION_ONLY % qid
  else:
    query = _BY_QID_QUERY % qid
  return query_api(query)

def query_api(query: Text) -> Sequence[Dict[Text, Any]]:
  '''Query the Wikidata API. 
  Arguments:
    queries(Text): The list of queries to run on the Wikidata API.
  Returns:
    The Wikidata items found as a Dictionary of:
    (1) head lables - {'vars': ['place', 'placeLabel', 'wikipediaUrl', 'point']}
    (2) results- e.g., {'place': {'type': 'uri', 'value':
    'http://www.wikidata....y/Q3272426'}, 'placeLabel': {'type': 'literal',
    'value': 'Equitable Life Building', 'xml:lang': 'en'}, 'point': {'type':
    'literal', 'value': 'Point(-74.010555555 ...708333333)'}, 'wikipediaUrl':
    {'type': 'uri', 'value': 'https://en.wikipedia...Manhattan)'}}
  '''

  endpoint_url = "https://query.wikidata.org/sparql"

  user_agent = "WDQS-example Python/%s.%s" % (
    sys.version_info[0], sys.version_info[1])
  sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
  sparql.setReturnFormat(JSON)
  sparql.setQuery(query)
  all_results = sparql.query().convert()

  return all_results["results"]["bindings"]
