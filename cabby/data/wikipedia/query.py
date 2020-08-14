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


def get_wikipedia_items(titles: Sequence[Text]) -> Dict:
    '''Query the Wikipedia API. 
    Arguments:
        title(Text): The wikipedia title to run on the Wikipedia API.
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
