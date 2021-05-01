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
'''
Define templates for RVS.
When adding a feature or landmark this file needs to be updated.
The update should be in adding terminals and\or rules.
The pivot or landmark should be the same as their name in
cabby.geo.walk.py file but with upper case.
E.g., 'end_point' should appear as 'END_POINT'.
'''

import flashtext
import inflect
from typing import Dict, Sequence, Text, Tuple
import nltk
from nltk import CFG, Production
from nltk.parse.generate import Nonterminal
import pandas as pd
import re

from cabby.geo import geo_item
from cabby.geo import walk

inflect_engine = inflect.engine()
STREET_FEATURES = ["next block", "next intersection", "blocks"] \
                  + walk.FEATURES_TYPES + walk.LANDMARK_TYPES

# Terminals in the grammar.
MAIN_NO_V = [
  "CARDINAL_DIRECTION from MAIN_PIVOT for INTERSECTIONS intersections",
  "CARDINAL_DIRECTION from MAIN_PIVOT for BLOCKS blocks",
  "CARDINAL_DIRECTION from MAIN_PIVOT, which will be your SPATIAL_REL_PIVOT, for BLOCKS blocks",
  "BLOCKS blocks past MAIN_PIVOT",
  "BLOCKS blocks past MAIN_PIVOT, which will be on your SPATIAL_REL_PIVOT",
  "INTERSECTIONS intersections past MAIN_PIVOT",
  "past MAIN_PIVOT",
  "past MAIN_PIVOT on your SPATIAL_REL_PIVOT",
  "CARDINAL_DIRECTION and past MAIN_PIVOT",
  "CARDINAL_DIRECTION and past MAIN_PIVOT (on your SPATIAL_REL_PIVOT)",
  "past MAIN_PIVOT and continue to the next intersection",
  "past MAIN_PIVOT on your SPATIAL_REL_PIVOT, and continue to the next intersection",
  "past MAIN_PIVOT, pass it on your SPATIAL_REL_PIVOT, and continue to the next intersection",
  "past MAIN_PIVOT and continue to the next block",
  "past MAIN_PIVOT, pass it on your SPATIAL_REL_PIVOT, and continue to the next block",
  "to MAIN_PIVOT and go CARDINAL_DIRECTION",
  "to MAIN_PIVOT, pass it on your SPATIAL_REL_PIVOT, and go CARDINAL_DIRECTION",
  "to MAIN_PIVOT and turn CARDINAL_DIRECTION",
  "to MAIN_PIVOT, pass it on your SPATIAL_REL_PIVOT, and turn CARDINAL_DIRECTION"

  ""
]

# SPATIAL_RELATION_PIVOT

MAIN = [
  ". When you pass MAIN_PIVOT, you'll be just INTERSECTIONS intersections away",
  ". When you pass MAIN_PIVOT on your SPATIAL_REL_PIVOT, you'll be just INTERSECTIONS intersections away",
  ". When you pass MAIN_PIVOT, you'll be just BLOCKS blocks away",
  ". When you pass MAIN_PIVOT on your SPATIAL_REL_PIVOT, you'll be just BLOCKS blocks away",
  ". MAIN_PIVOT is INTERSECTIONS intersections away",
  ". MAIN_PIVOT is INTERSECTIONS intersections away on the SPATIAL_REL_PIVOT side of the street",
  ". MAIN_PIVOT is BLOCKS blocks away",
  ". MAIN_PIVOT is BLOCKS blocks away, on the SPATIAL_REL_PIVOT side of the street,",
  ". Head to MAIN_PIVOT and go INTERSECTIONS intersections further",
  ". Head to MAIN_PIVOT. Pass MAIN_PIVOT on your SPATIAL_REL_PIVOT and go INTERSECTIONS intersections further",
  ". Go to MAIN_PIVOT and walk INTERSECTIONS intersections past it",
  ". Go to MAIN_PIVOT, pass it on your SPATIAL_REL_PIVOT, and walk INTERSECTIONS intersections past it",
  ". Travel to MAIN_PIVOT and continue BLOCKS blocks further",
  ". Walk to MAIN_PIVOT and proceed BLOCKS blocks past it",
  ". After you reach MAIN_PIVOT, you'll need to go INTERSECTIONS intersections further",
  ". After you reach MAIN_PIVOT on your SPATIAL_REL_PIVOT, "+
  "you'll need to go INTERSECTIONS intersections further",
  ". When you get to MAIN_PIVOT, you have INTERSECTIONS intersections more to walk",
  ". When you see to MAIN_PIVOT on your SPATIAL_REL_PIVOT, "+
  "you have INTERSECTIONS intersections more to walk",
  ". After you pass MAIN_PIVOT, go BLOCKS blocks more",
  ". After you pass MAIN_PIVOT on your SPATIAL_REL_PIVOT, go BLOCKS blocks more",
  ". Once you reach MAIN_PIVOT, continue for BLOCKS blocks",
  ". Once you see MAIN_PIVOT on your SPATIAL_REL_PIVOT, continue for BLOCKS blocks",

]

V1 = ['Go', 'Walk', 'Head', 'Proceed', 'Travel']

V2 = [
  'Meet at the END_POINT.',
  'Come to the END_POINT.',
  'Head over to the END_POINT.',
  'The END_POINT is the meeting point.'
]

NEAR_GOAL_END = [
  ". The END_POINT will be near a NEAR_PIVOT.",
  ". The END_POINT will be on your SPATIAL_REL_GOAL, near a NEAR_PIVOT.",
  ". A NEAR_PIVOT is quite close to the END_POINT.",
  ". Meet at the END_POINT, which is right next to a NEAR_PIVOT.",
  ". Meet at the END_POINT, which will be on your SPATIAL_REL_GOAL, right next to a NEAR_PIVOT.",
  ". If you see a NEAR_PIVOT, you should find the END_POINT close by.",
  ". If you see a NEAR_PIVOT, you should find on the SPATIAL_REL_GOAL side of the street"+
  " the END_POINT close by."

]
NEAR_GOAL_START = [
  ". It will be near a NEAR_PIVOT.",
  ". It will be on your SPATIAL_REL_GOAL, near a NEAR_PIVOT.",
  ". It is close to a NEAR_PIVOT.",
  ". It is on the SPATIAL_REL_GOAL side of the street, close to a NEAR_PIVOT.",
  ". A NEAR_PIVOT is close by."
]
AVOID = [
  ". If you reach BEYOND_PIVOT, you have gone too far.",
  ". You've overshot the meeting point if you reach BEYOND_PIVOT.",
  ". If you pass BEYOND_PIVOT, you've gone too far.",
  ""]

GOAL_END = [
  'and meet at the END_POINT, right GOAL_POSITION.',
  'and meet at the END_POINT.',
  'and come to the END_POINT, right GOAL_POSITION.',
  'and come to the END_POINT.',
  'to reach the END_POINT, right GOAL_POSITION.',
  'to reach the END_POINT.',
  'to arrive at the END_POINT, right GOAL_POSITION.'
  'to arrive at the END_POINT.'
]

GOAL = ["the END_POINT"]


def add_rules(nonterminal_name: Text,
        list_terminals: Sequence[Text]) -> Sequence[Production]:
  """Create the production rules for a givn nonterminal and a
   list of terminals corresponding to it.
  Arguments:
    nonterminal_name: The name of the nonterminal.
    list_terminals: The list of terminals that for each one a rule with 
    the nonterminal will be produced.
  Returns:
    A sequence of productions rules.
  """
  prods = []
  for phrase in list_terminals:
    rule = Production(Nonterminal(nonterminal_name), (phrase,))
    prods.append(rule)
  return prods


def create_templates():
  """Creates the templates from the grammar."""

  prods = [
    # Specific verb with goal and the rest of instruction body.
    Production(Nonterminal('S'), (Nonterminal('V2'),
                    Nonterminal('V2_BODY'))),
    # A verb and rest of the instruction body assuming goal already mentioned.
    Production(Nonterminal('V2_BODY'), (Nonterminal('V1'),
                      Nonterminal('M_G_ALREADY_V'))),
    # A verb and the rest of the instruction body assuming the goal wasn't
    # mentioned before.
    Production(Nonterminal('S'), (Nonterminal(
      'V1'), Nonterminal('NO_GOAL'))),
    # The goal in the begining and the rest of the instruction body assuming
    # goal already mentioned.
    Production(Nonterminal('S'), (Nonterminal('V1_GOAL'),
                    Nonterminal('WITH_GOAL'))),
    # A verb and 'to the' and then goal mention and the rest of the instruction
    # body.
    Production(Nonterminal('V1_GOAL'), (Nonterminal('V1'),
                      Nonterminal('V1_CON'))),
    # A goal mention and the rest of the instruction body.
    Production(Nonterminal('WITH_GOAL'), (Nonterminal('GOAL'),
                        Nonterminal('M_G'))),
    # Main part of the instruction without verb in begining and resuming
    # sentence.
    Production(Nonterminal('M_G_ALREADY_V'), (Nonterminal('MAIN_NO_V'),
                    Nonterminal('END_NEAR_GOAL_KNOWN'))),
    # # Main part of the instruction, adding a new sentence.
    Production(Nonterminal('M_G'), (Nonterminal('MAIN'),
                    Nonterminal('END_NEAR_GOAL_KNOWN'))),
    # End part - (1) near pivot assuming goal already mentioned; and (2) avoid
    # sentence.
    Production(
      Nonterminal('END_NEAR_GOAL_KNOWN'), (Nonterminal('NEAR_GOAL_START'),
                    Nonterminal('AVOID'))),
    # End part - (1) near pivot assuming goal not mentioned yet; and (2) avoid
    # sentence.
    Production(Nonterminal(
      'END_NEAR_GOAL_KNOWN'), (Nonterminal('NEAR_GOAL_END'),
                    Nonterminal('AVOID'))),
    # Main part of the instruction without verb in begining and resuming
    # sentence assuming no goal mentioned before.
    Production(Nonterminal('NO_GOAL'), (Nonterminal('MAIN_NO_V'),
                      Nonterminal('END_NEAR_GOAL_UNKNOWN'))),
    # Add Goal to main part and then resume instruction by adding an
    # ending(near+avoid).
    Production(Nonterminal('END_NEAR_GOAL_UNKNOWN'), (Nonterminal('GOAL_END'),
                    Nonterminal('END_NEAR_GOAL_KNOWN'))),
    # Add Goal with near and then add an avoid sentenece.
    Production(
      Nonterminal('END_NEAR_GOAL_UNKNOWN'), (Nonterminal('NEAR_GOAL_END'),
                    Nonterminal('AVOID'))),
    # Termial for IN+DT after verb.
    Production(Nonterminal('V1_CON'), ('to the',)),

  ]

  prods += add_rules('V2', V2)
  prods += add_rules('AVOID', AVOID)
  prods += add_rules('NEAR_GOAL_START', NEAR_GOAL_START)
  prods += add_rules('NEAR_GOAL_END', NEAR_GOAL_END)
  prods += add_rules('GOAL', GOAL)
  prods += add_rules('GOAL_END', GOAL_END)
  prods += add_rules('MAIN_NO_V', MAIN_NO_V)
  prods += add_rules('MAIN', MAIN)
  prods += add_rules('V1', V1)

  grammar = CFG(Nonterminal('S'), prods)

  # Generate templates.
  templates = []
  for sentence in nltk.parse.generate.generate(grammar):

    sentence = ' '.join(sentence)

    if sentence[-1] != '.':
      sentence += '.'
    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace("..", ".")

    re_space = re.compile(r'[\s]+')
    sentence = re_space.sub(r' ', sentence)

    templates.append(sentence)

  templates_df = pd.DataFrame(
    templates, columns=['sentence']).drop_duplicates()
  # Save templates
  templates_df.to_csv('templates.csv', index=False, header = False)

  # Flag features.
  for column in STREET_FEATURES:
    templates_df[column] = templates_df['sentence'].apply(
    lambda x: column.upper() in x)

  return templates_df


def add_features_to_template(template: Text, entity: geo_item.GeoEntity) -> Tuple[Text, Dict[str, Tuple[int, int]]]:
  '''Add the entity features to the picked template to create an instruction.
  Arguments:
    template: The chosen template.
    entity: The features of the path to add to the template.
  Returns:
    The instruction created and a dictionary of the landmarks(keys)
    and spans(values) in the instruction.
  '''

  intersections = entity.geo_features['intersections']
  intersections = -1 if not intersections else int(intersections)
  blocks = str(intersections-1)

  if entity.geo_landmarks['end_point'].main_tag[0].isupper():
    template = template.replace('The END_POINT', 'END_POINT')
    template = template.replace('the END_POINT', 'END_POINT')
  near_landmark = entity.geo_landmarks['near_pivot'].main_tag
  if near_landmark[0].isupper():
    template = template.replace('A NEAR_PIVOT', 'NEAR_PIVOT')
    template = template.replace('a NEAR_PIVOT', 'NEAR_PIVOT')
  if inflect_engine.singular_noun(near_landmark):
    template = template.replace('a NEAR_PIVOT', 'NEAR_PIVOT')
    template = template.replace('A NEAR_PIVOT', '?UP?NEAR_PIVOT')
    template = template.replace('NEAR_PIVOT is', 'NEAR_PIVOT are')
  template = template.replace('BLOCKS', blocks)

  entities_tags = []

  for landmark_type, landmark in entity.geo_landmarks.items():

    if landmark.main_tag:

      template = template.replace("?UP?"+landmark_type.upper(),
                                  landmark.main_tag.capitalize())


      template = template.replace(landmark_type.upper(),
                                  landmark.main_tag)

      if landmark.landmark_type != "start_point":
        entities_tags.append(landmark.main_tag)


  for feature_type, feature in entity.geo_features.items():

    template = template.replace(feature_type.upper(),
                                str(feature))

  # Fix text.
  template = template.replace('The The', 'The')
  template = template.replace('the The', 'the')

  re_indef_vowel  = re.compile(r'\ba ([aeiou])')
  template = re_indef_vowel.sub(r'an \1', template)

  entities_span_dict = {}
  for entity_tag in entities_tags:
    entities_span_dict.update(add_entity_span(entity_tag, template))

  template = template.replace('_', ' ')

  return template, entities_span_dict

def add_entity_span(entity_tag: str, instruction: str) -> Dict[str, Tuple[int, int]]:
  '''Adds the entity span to a dictionary of entites (keys) and spans (value).
    Args:
      entity_tag: The entity tag name.
      instruction: instruction with entity.
    Returns:
      A dictionary of entities (keys) and spans (values).
  '''

  keyword_processor = flashtext.KeywordProcessor(case_sensitive=False)
  keyword_processor.add_keyword(entity_tag)

  keywords_found = keyword_processor.extract_keywords(instruction, span_info=True)

  entities_span_dict = {}
  for keyword_found in keywords_found:
    start_position, end_position = keyword_found[1], keyword_found[2]
    entities_span_dict[entity_tag] = (start_position, end_position)

  return entities_span_dict
