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

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Dict, Tuple, Sequence, Text, Any

from cabby.geo import util
from cabby.geo import regions


_MANHATTAN_QUERY = """SELECT ?place ?placeLabel ?wikipediaUrl
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
            bd:serviceParam wikibase:cornerWest "Point(-74.0379,40.6966)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-73.9293,40.7963)"^^geo:wktLiteral .
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
            bd:serviceParam wikibase:cornerWest "Point(-74.0379,40.6966)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-73.9293,40.7963)"^^geo:wktLiteral .
            }
          }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

          }


          GROUP BY ?place ?placeLabel ?wikipediaUrl 
        """

_PITTSBURGH_QUERY = """SELECT ?place ?placeLabel ?wikipediaUrl
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
            bd:serviceParam wikibase:cornerWest "Point(-80.035,40.425)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-79.930,40.460)"^^geo:wktLiteral .
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
            bd:serviceParam wikibase:cornerWest "Point(-80.035,40.425)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-79.930,40.460)"^^geo:wktLiteral .
            }
          }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

          }

          GROUP BY ?place ?placeLabel ?wikipediaUrl 
        """

_DC_QUERY = """SELECT ?place ?placeLabel ?wikipediaUrl
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
            bd:serviceParam wikibase:cornerWest "Point(-77.04053,38.90821)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-77.03937,38.90922)"^^geo:wktLiteral .
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
            bd:serviceParam wikibase:cornerWest "Point(-77.04053,38.90821)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-77.03937,38.90922)"^^geo:wktLiteral .
            }
          }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

          }


          GROUP BY ?place ?placeLabel ?wikipediaUrl 
        """


_BY_QID_QUERY = """SELECT ?place ?placeLabel ?wikipediaUrl
           ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
          (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
          WHERE 
          {
          {
            VALUES ?place {wd:%s}
            ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
            bd:serviceParam wikibase:cornerWest "Point(-80.035,40.425)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-72.963,41.752)"^^geo:wktLiteral .
            }
          }
          FILTER (?instance  not in
          (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
          wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
          wd:Q744913, wd:Q186117, wd:Q3298291) )
          }
          GROUP BY ?place ?placeLabel ?wikipediaUrl 
        """


def get_geofenced_wikidata_items(region: Text) -> Sequence[Dict[Text, Any]]:

  '''Get Wikidata items for a specific area. 
  Arguments:
    region(Text): The area to query the Wikidata items.
  Returns:
    The Wikidata items found in the area.
  '''

  if region == "Pittsburgh":
    query = _PITTSBURGH_QUERY

  elif region == "Manhattan":
    query = _MANHATTAN_QUERY
  
  elif region == "DC":
    query = _DC_QUERY
    
  else:
    raise ValueError(f"{region} is not a supported region.")

  results = query_api(query)

  # Filter by map region.
  polygon_region = regions.get_region(region)
  filtered_results = []
  for result in results:
    point_str = result['point']['value']
    point = util.point_str_to_shapely_point(point_str) 
    if polygon_region.contains(point):
      filtered_results.append(result)
  return filtered_results



def get_geofenced_wikidata_items_by_qid(qid: Text) -> Sequence[Dict[Text, Any]]:
  '''Get Wikidata items for a specific area. 
  Arguments:
    qid(Text): The qid to query the Wikidata items.
  Returns:
    The Wikidata items found in the area.
  '''
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