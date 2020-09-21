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
'''Define templates for RVS.'''

from nltk import CFG, Production
from nltk.parse.generate import generate, Nonterminal
import pandas as pd
from typing import Dict, Sequence, Text


# Terminals in the grammar.
MAIN_NO_V = [
    "CARDINAL_DIRECTION from MAIN_PIVOT for NUMBER_INTERSECTIONS intersections",
    "CARDINAL_DIRECTION from MAIN_PIVOT for NUMBER_BLOCKS blocks",
    "NUMBER_BLOCKS blocks past MAIN_PIVOT",
    "NUMBER_INTERSECTIONS intersections past MAIN_PIVOT",
    "past MAIN_PIVOT",
    "CARDINAL_DIRECTION past MAIN_PIVOT"
    "next intersection past MAIN_PIVOT",
    "next block past MAIN_PIVOT",
]
MAIN = [
    ". When you pass MAIN_PIVOT, you'll be just NUMBER_INTERSECTIONS intersections away",
    ". When you pass MAIN_PIVOT, you'll be just NUMBER_BLOCKS blocks away",
]
V1 = ['Go', 'Walk', 'Head']
V2 = ['Meet at GOAL_LOCATION.', 'Come to GOAL_LOCATION.']
NEAR_GOAL_END = [". GOAL_LOCATION will be near NEAR_PIVOT."]
NEAR_GOAL_START = [". It will be near NEAR_PIVOT."]
AVOID = [
        '. If you reach BEYOND_PIVOT, you have gone too far.',
         '']
GOAL_END = ['and meet at GOAL_LOCATION.']
GOAL = ["GOAL_LOCATION"]


def add_rules(nonterminal_name: Text, list_terminals: Sequence[Text]) -> Sequence[Production]:
    """Create the production rules for a givn nonterminal and a list of terminals corresponding to it.

    Args:
      nonterminal_name: The name of the nonterminal.
      list_terminals: The list of terminals that for each one a rule with the nonterminal will be produced.
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
      Production(Nonterminal('S'), (Nonterminal('V2'), Nonterminal('V2_MAIN'))),
      Production(Nonterminal('V2_MAIN'),
                (Nonterminal('V1'), Nonterminal('M_G'))),

      Production(Nonterminal('S'), (Nonterminal('V1'), Nonterminal('NO_GOAL'))),
      Production(Nonterminal('S'), (Nonterminal(
          'V1_GOAL'), Nonterminal('WITH_GOAL'))),
      Production(Nonterminal('V1_GOAL'),
                (Nonterminal('V1'), Nonterminal('V1_CON'))),
      Production(Nonterminal('WITH_GOAL'),
                (Nonterminal('GOAL'), Nonterminal('M_G'))),
      Production(Nonterminal('M_G'),

                (Nonterminal('MAIN_NO_V'), Nonterminal('E'))),
      Production(Nonterminal('M_G'), (Nonterminal('MAIN'), Nonterminal('E'))),
      Production(Nonterminal('E'), (Nonterminal(
          'NEAR_GOAL_START'), Nonterminal('AVOID'))),    
      Production(Nonterminal('E'), (Nonterminal(
          'NEAR_GOAL_END'), Nonterminal('AVOID'))),
      Production(Nonterminal('V1_CON'), ('to the',)),
      Production(Nonterminal('NO_GOAL'),
                (Nonterminal('MAIN_NO_V'), Nonterminal('G'))),
      Production(Nonterminal('G'), (Nonterminal(
          'GOAL_END'), Nonterminal('NEAR_GOAL_END'))),
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
  for sentence in generate(grammar):

    sentence = ' '.join(sentence)
    if sentence[-1] != '.':
        sentence += '.'
    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace("..", ".")
    templates.append(sentence)

  templates_df = pd.DataFrame(templates, columns=['sentence'])

  # Flag features.
  templates_df['cardinal_direction'] = templates_df['sentence'].apply(
      lambda x: 'CARDINAL_DIRECTION' in x)
  templates_df['blocks'] = templates_df['sentence'].apply(
      lambda x: 'NUMBER_BLOCKS' in x)
  templates_df['intersection'] = templates_df['sentence'].apply(
      lambda x: 'NUMBER_INTERSECTIONS' in x)
  templates_df['beyond_pivot'] = templates_df['sentence'].apply(
      lambda x: 'BEYOND_PIVOT' in x)
  templates_df['next_block'] = templates_df['sentence'].apply(
      lambda x: 'next block' in x)
  templates_df['next_intersection'] = templates_df['sentence'].apply(
      lambda x: 'next intersection' in x)

  return templates_df

def add_features_to_template(template, goal, main_pivot, near_pivot, beyond_pivot, intersection, blocks, cardinal_direction):
  template= template.replace('GOAL_LOCATION', goal)
  template= template.replace('MAIN_PIVOT', main_pivot)
  template= template.replace('NEAR_PIVOT', near_pivot)
  template= template.replace('BEYOND_PIVOT', beyond_pivot)
  template= template.replace('NUMBER_INTERSECTIONS', str(int(intersection)))
  template= template.replace('NUMBER_BLOCKS', str(int(blocks)))
  template= template.replace('CARDINAL_DIRECTION', cardinal_direction)

  # Fixparesing.
  template = template.replace('The The', 'The')
  template = template.replace('the The', 'the')
  return template




# print (create_templates()['sentence'].values)


