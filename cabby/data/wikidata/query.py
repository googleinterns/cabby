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

'''Library to support Wikidata areal queries.'''

import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from typing import Dict, Tuple, Sequence, Text

_MANHATTAN_QUERY = """SELECT ?place ?placeLabel (GROUP_CONCAT(DISTINCT
            ?location;separator=", ") AS ?point)   Where {{SELECT  ?place
            ?placeLabel ?instance ?instanceLabel ?placeAltLabel
            ?placeDescription ?location WHERE 
            {
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.028,40.705)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.975,40.712)"^^geo:wktLiteral .
            }
                }
                UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.02,40.695)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-74,40.705)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.028,40.712)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.973,40.718)"^^geo:wktLiteral .
            }
                }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.028,40.718)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.972,40.725)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.028,40.712)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.973,40.718)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.02,40.725)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.972,40.73)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.02,40.73)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.969,40.738)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.02,40.738)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.969,40.745)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.013,40.745)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.963,40.752)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.013,40.752)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.959,40.76)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74.01,40.76)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.952,40.7645)"^^geo:wktLiteral .
            }
            }
            UNION
            {
            ?place wdt:P31 ?instance. SERVICE wikibase:label { bd:serviceParam
            wikibase:language "[AUTO_LANGUAGE],en". } SERVICE wikibase:box
            {?place wdt:P625 ?location . bd:serviceParam wikibase:cornerWest
            "Point(-74,40.7645)"^^geo:wktLiteral . bd:serviceParam
            wikibase:cornerEast "Point(-73.946,40.772)"^^geo:wktLiteral .
            }
            }
            }
            }
                FILTER (?instance  not in (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757, wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683, wd:Q744913          ) )
            }
            GROUP BY ?place ?placeLabel
            """

_PITTSBURGH_QUERY = """SELECT ?place ?placeLabel
                    (GROUP_CONCAT(DISTINCT?location;separator=", ") AS ?point)
                    WHERE 
                    {
                    {
                        ?place wdt:P31 ?instance.
                        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
                        SERVICE wikibase:box {
                        ?place wdt:P625 ?location .
                        bd:serviceParam wikibase:cornerWest "Point(-80.035,40.425)"^^geo:wktLiteral .
                        bd:serviceParam wikibase:cornerEast "Point(-79.930,40.460)"^^geo:wktLiteral .
                        }
                    }
                    FILTER (?instance  not in
                    (wd:Q34442,wd:Q12042110,wd:Q124757,wd:Q79007,wd:Q18340514,wd:Q537127,wd:Q1311958,wd:Q124757,
                    wd:Q25917154,  wd:Q1243306, wd:Q1570262, wd:Q811683,
                    wd:Q744913          ) )
                    }
                    GROUP BY ?place ?placeLabel
                """


def get_geofenced_wikidata_items(region: Text) -> Dict:
    '''Get Wikidata items for a specific area. 
    Arguments:
        region(Text): The area to query the Wikidata items.
    Returns:
        The Wikidata items found in the area.
    '''

    assert region == "Manhattan" or region == "Pittsburgh"
    if region == "Pittsburgh":
        query = _PITTSBURGH_QUERY

    elif region == "Manhattan":
        query = _MANHATTAN_QUERY
    else:
        raise ValueError(f"{region} is not a supported region.")

    return query_api(query)


def query_api(query: Text) -> Dict:
    '''Get Wikidata query. 
    Arguments:
        query(Text): The query to run on the Wikidata api.
    Returns:
        The Wikidata items found.
    '''

    endpoint_url = "https://query.wikidata.org/sparql"

    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()
