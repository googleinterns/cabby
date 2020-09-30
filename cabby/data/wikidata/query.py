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
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Dict, Tuple, Sequence, Text

_MANHATTAN_QUERY = ["""SELECT ?place ?placeLabel ?wikipediaUrl 
      ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
      (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)   Where
      {{SELECT  ?place?placeLabel ?instance ?instanceLabel ?wikipediaUrl?location WHERE 
      {
      {
      ?place wdt:P31 ?instance.?wikipediaUrl
      schema:about?place.?wikipediaUrl schema:isPartOf
      <https://en.wikipedia.org/>. SERVICE wikibase:label {bd:serviceParam
      wikibase:language "[AUTO_LANGUAGE],en". } 
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
      SERVICE wikibase:box
      {?place wdt:P625?location . bd:serviceParam wikibase:cornerWest
      "Point(-74.028,40.705)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-73.975,40.712)"^^geo:wktLiteral .
      }
        }
        UNION
      {
                  ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
        SERVICE wikibase:label { bd:serviceParam wikibase:language
      "[AUTO_LANGUAGE],en". } 
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
      SERVICE wikibase:box {?place
      wdt:P625?location . bd:serviceParam wikibase:cornerWest
      "Point(-74.02,40.695)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-74,40.705)"^^geo:wktLiteral .
      }
      }
      UNION
      {
       ?place wdt:P31 ?instance.?wikipediaUrl
       schema:about?place.?wikipediaUrl schema:isPartOf
       <https://en.wikipedia.org/>. SERVICE wikibase:label
       {bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
       ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
       SERVICE
       wikibase:box {?place wdt:P625?location . bd:serviceParam
       wikibase:cornerWest "Point(-74.028,40.712)"^^geo:wktLiteral .
       bd:serviceParam wikibase:cornerEast
       "Point(-73.973,40.718)"^^geo:wktLiteral .
      }
        }
      UNION
      {
      ?place wdt:P31 ?instance.?wikipediaUrl
      schema:about?place.?wikipediaUrl schema:isPartOf
      <https://en.wikipedia.org/>. SERVICE wikibase:label {bd:serviceParam
      wikibase:language "[AUTO_LANGUAGE],en". } 
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
      SERVICE wikibase:box
      {?place wdt:P625?location . bd:serviceParam wikibase:cornerWest
      "Point(-74.028,40.718)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-73.972,40.725)"^^geo:wktLiteral .
      }
      }
      UNION
      {
      ?place wdt:P31 ?instance.?wikipediaUrl
      schema:about?place.?wikipediaUrl schema:isPartOf
      <https://en.wikipedia.org/>. SERVICE wikibase:label {bd:serviceParam
      wikibase:language "[AUTO_LANGUAGE],en". } 
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
      SERVICE wikibase:box
      {?place wdt:P625?location . bd:serviceParam wikibase:cornerWest
      "Point(-74.028,40.712)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-73.973,40.718)"^^geo:wktLiteral .
      }
      }
      UNION
      {
       ?place wdt:P31 ?instance.?wikipediaUrl
       schema:about?place.?wikipediaUrl schema:isPartOf
       <https://en.wikipedia.org/>. SERVICE wikibase:label
       {bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
       ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
       SERVICE
       wikibase:box {?place wdt:P625?location . bd:serviceParam
       wikibase:cornerWest "Point(-74.02,40.725)"^^geo:wktLiteral .
       bd:serviceParam wikibase:cornerEast
       "Point(-73.972,40.73)"^^geo:wktLiteral .
      }
      }

      }
      }
        FILTER (?instance  not in (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757, wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683, wd:Q744913, wd:Q186117, wd:Q3298291) )
      }
      GROUP BY ?place ?placeLabel ?wikipediaUrl ?instanceLabel
        """,

          """SELECT ?place ?placeLabel ?wikipediaUrl 
      ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
      (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)   Where
      {{SELECT  ?place?placeLabel ?instance ?instanceLabel ?wikipediaUrl?location WHERE 
      {

      {
       ?place wdt:P31 ?instance.?wikipediaUrl
       schema:about?place.?wikipediaUrl schema:isPartOf
       <https://en.wikipedia.org/>. SERVICE wikibase:label
       {bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
       ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
       SERVICE
       wikibase:box {?place wdt:P625?location . bd:serviceParam
       wikibase:cornerWest "Point(-74.02,40.73)"^^geo:wktLiteral .
       bd:serviceParam wikibase:cornerEast
       "Point(-73.969,40.738)"^^geo:wktLiteral .
      }
      }
      UNION
      {
       ?place wdt:P31 ?instance.?wikipediaUrl
       schema:about?place.?wikipediaUrl schema:isPartOf
       <https://en.wikipedia.org/>. SERVICE wikibase:label
       {bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
       ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
       SERVICE
       wikibase:box {?place wdt:P625?location . bd:serviceParam
       wikibase:cornerWest "Point(-74.02,40.738)"^^geo:wktLiteral .
       bd:serviceParam wikibase:cornerEast
       "Point(-73.969,40.745)"^^geo:wktLiteral .
      }
      }
      UNION
      {
                  ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>.  SERVICE wikibase:label { bd:serviceParam
      wikibase:language "[AUTO_LANGUAGE],en". } 
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
      SERVICE wikibase:box
      {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
      "Point(-74.013,40.745)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-73.963,40.752)"^^geo:wktLiteral .
      }
      }
      UNION
      {
                  ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>.  SERVICE wikibase:label { bd:serviceParam
      wikibase:language "[AUTO_LANGUAGE],en". } 
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
      SERVICE wikibase:box
      {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
      "Point(-74.013,40.752)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-73.959,40.76)"^^geo:wktLiteral .
      }
      }
      UNION
      {
      ?place wdt:P31 ?instance.?wikipediaUrl
      schema:about?place.?wikipediaUrl schema:isPartOf
      <https://en.wikipedia.org/>. SERVICE wikibase:label {bd:serviceParam
      wikibase:language "[AUTO_LANGUAGE],en". } 
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
      SERVICE wikibase:box
      {?place wdt:P625?location . bd:serviceParam wikibase:cornerWest
      "Point(-74.01,40.76)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-73.952,40.7645)"^^geo:wktLiteral .
      }
      }
      UNION
      {
        ?place wdt:P31 ?instance.?wikipediaUrl
        schema:about?place.?wikipediaUrl schema:isPartOf
        <https://en.wikipedia.org/>. 

       SERVICE wikibase:label { bd:serviceParam wikibase:language
      "[AUTO_LANGUAGE],en". }
      ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
       SERVICE wikibase:box {?place
      wdt:P625?location . bd:serviceParam wikibase:cornerWest
      "Point(-74,40.7645)"^^geo:wktLiteral . bd:serviceParam
      wikibase:cornerEast "Point(-73.946,40.772)"^^geo:wktLiteral .
      }
      }

      }
      }
        FILTER (?instance  not in (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757, wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683, wd:Q744913, wd:Q186117, wd:Q3298291 ) )
      }
      GROUP BY ?place ?placeLabel ?wikipediaUrl
      """

          ]

_PITTSBURGH_QUERY = ["""SELECT ?place ?placeLabel ?wikipediaUrl
           ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
          (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
          WHERE 
          {
          {
            ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").

            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
            bd:serviceParam wikibase:cornerWest "Point(-80.035,40.425)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-79.930,40.460)"^^geo:wktLiteral .
            }
          }
          FILTER (?instance  not in
          (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
          wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
          wd:Q744913, wd:Q186117, wd:Q3298291) )
          }
          GROUP BY ?place ?placeLabel ?wikipediaUrl 
        """]

_PITTSBURGH_RELATION_QUERY = ["""SELECT ?place ?placeLabel ?p ?propLabel ?instance ?instanceLabel
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
            # SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } 
              ?prop wikibase:directClaim ?p .
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").

            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
            bd:serviceParam wikibase:cornerWest "Point(-80.035,40.425)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-79.930,40.460)"^^geo:wktLiteral .
            }
          }
          FILTER (?instance  not in
          (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
          wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
          wd:Q744913, wd:Q186117, wd:Q3298291) )
          }
        """]

_DC_QUERY = ["""SELECT ?place ?placeLabel ?wikipediaUrl
           ( GROUP_CONCAT ( DISTINCT ?instanceLabel; separator="; " ) AS ?instance )
          (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
          WHERE 
          {
          {
            ?place wdt:P31 ?instance.
            ?wikipediaUrl schema:about ?place. 
            ?wikipediaUrl schema:isPartOf <https://en.wikipedia.org/>. 
            SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        ?instance rdfs:label ?instanceLabel.  filter(lang(?instanceLabel) = "en").
            SERVICE wikibase:box {
            ?place wdt:P625 ?location .
            bd:serviceParam wikibase:cornerWest "Point(-77.04053,38.90821)"^^geo:wktLiteral .
            bd:serviceParam wikibase:cornerEast "Point(-77.03937,38.90922)"^^geo:wktLiteral .
            }
          }
          FILTER (?instance  not in
          (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
          wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
          wd:Q744913, wd:Q186117, wd:Q3298291) )
          }
          GROUP BY ?place ?placeLabel ?wikipediaUrl 
        """]

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


def get_geofenced_wikidata_items(region: Text) -> Sequence[Dict]:
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
  
  elif region == "D.C":
    query = _DC_QUERY
    
  else:
    raise ValueError(f"{region} is not a supported region.")

  return query_api(query)


def get_geofenced_wikidata_relations(region: Text,
                                     extract_qids = False) -> Sequence[Dict]:
  '''Get Wikidata relations for a specific area.
  Arguments:
    region(Text): The area to query the Wikidata items.
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
  if region == "Pittsburgh":
    query = _PITTSBURGH_RELATION_QUERY

  elif region == "Manhattan":
    raise NotImplementedError(f"{region} is not an implemented region.")
  
  elif region == "D.C":
    raise NotImplementedError(f"{region} is not an implemented region.")
    
  else:
    raise ValueError(f"{region} is not a supported region.")

  query_result = query_api(query)
  result_df = pd.DataFrame([{k: v['value'] for k, v in x.items()} for x in query_result])
  if extract_qids:
    extract_qid = lambda s: s.apply(lambda x: x.rsplit("/", 1)[1])
    extract_cols = ["place", "p", "instance"]
    result_df[extract_cols] = result_df[extract_cols].apply(extract_qid)
  return result_df

def get_geofenced_wikidata_items_by_qid(qid: Text) -> Sequence[Dict]:
  '''Get Wikidata items for a specific area. 
  Arguments:
    qid(Text): The qid to query the Wikidata items.
  Returns:
    The Wikidata items found in the area.
  '''
  query = _BY_QID_QUERY % qid

  return query_api([query])


def query_api(queries: Sequence[Text]) -> Dict[Text, Dict]:
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
  sparql.setQuery(queries[0])
  all_results = sparql.query().convert()

  for query in queries[1:]:
    sparql.setQuery(query)
    query_results = sparql.query().convert()
    all_results['results']['bindings'] += query_results['results']['bindings']

  return all_results["results"]["bindings"]
