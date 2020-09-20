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
'''Define grammar for RVS templates'''

from nltk import CFG, Production
from nltk.parse.generate import generate, Nonterminal



def add_rules(nonterminal_name, list_terminals):
  prods = []
  for phrase in list_terminals:
    rule = Production(Nonterminal(nonterminal_name), (phrase,))
    prods.append(rule)
  return prods

MAIN_NO_V = [
", CARDINAL_DIRECTION from MAIN_PIVOT for NUMBER_INTERSECTIONS intersections",
", CARDINAL_DIRECTION from MAIN_PIVOT for NUMBER_BLOCKS blocks",
", NUMBER_BLOCKS blocks past MAIN_PIVOT",
", NUMBER_INTERSECTIONS intersections past MAIN_PIVOT",
", past MAIN_PIVOT",
", CARDINAL_DIRECTION past MAIN_PIVOT"
", NEXT intersection past MAIN_PIVOT",
", NEXT block past MAIN_PIVOT",
]


MAIN = [
". When you pass MAIN_PIVOT, you'll be just NUMBER_INTERSECTIONS intersections away", 
". When you pass MAIN_PIVOT, you'll be just NUMBER_BLOCKS blocks away", 
]


V1 = ['Go', 'Walk', 'Come', 'Head' ]
V2 = ['Meet']
GOAL_END = ['and meet at GOAL.']
NEAR_GOAL_END = ["The GOAL will be near NEAR_PIVOT."]
NEAR_GOAL_START = [". It will be near NEAR_PIVOT."]
AVOID = ['. If you reached BEYOND_PIVOT, you have gone too far.',
'']
GOAL = ["GOAL"]

prods = [
Production(Nonterminal('S'), (Nonterminal('V1'), Nonterminal('NO_GOAL'))), 
Production(Nonterminal('S'), (Nonterminal('V1_GOAL'), Nonterminal('WITH_GOAL'))), 
Production(Nonterminal('V1_GOAL'), (Nonterminal('V1'), Nonterminal('V1_CON'))), 


Production(Nonterminal('WITH_GOAL'), (Nonterminal('GOAL'), Nonterminal('M_G'))), 



Production(Nonterminal('M_G'), (Nonterminal('MAIN_NO_V'), Nonterminal('E'))), 
Production(Nonterminal('M_G'), (Nonterminal('MAIN'), Nonterminal('E'))), 

Production(Nonterminal('E'), (Nonterminal('NEAR_GOAL_START'), Nonterminal('AVOID'))), 


Production(Nonterminal('GOAL'), ('GOAL',)),
Production(Nonterminal('V1_CON'), ('to the',)),


Production(Nonterminal('NO_GOAL'), (Nonterminal('MAIN_NO_V'), Nonterminal('G'))),
Production(Nonterminal('G'), (Nonterminal('GOAL_END'), Nonterminal('NEAR_GOAL_END'))),
         ]


prods += add_rules('AVOID', AVOID)         
prods += add_rules('NEAR_GOAL_START', NEAR_GOAL_START)
prods += add_rules('NEAR_GOAL_END', NEAR_GOAL_END)
prods += add_rules('GOAL_END', GOAL_END)
prods += add_rules('MAIN_NO_V', MAIN_NO_V)
prods += add_rules('MAIN', MAIN)
prods += add_rules('V1', V1)


rule = Production(Nonterminal('E'), ('',))
prods.append(rule)


grammar = CFG(Nonterminal('S'), prods)
# grammar = CFG.fromstring("""
#   S -> V1 MAIN_PART_NO_GOAL
#   MAIN_PART_NO_GOAL -> 'X INTERSECTIONS CARDINAL_DIRECTION' | 'NUMBER_INTERSECTIONS intersections past MAIN_PIVOT.'
#   V1 -> 'Go' | 'Walk' | 'Come' | 'Head' 
#   V2 -> 'Meet'
# """)

counter=0
for sentence in generate(grammar, n=200):
    counter+=1

    sentence = ' '.join(sentence)
    if sentence[-1]!='.':
      sentence += '.'
    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace("..", ".")


    print (sentence)
print (counter)
# print('A Grammar:', grammar)
# print('grammar.start()   =>', grammar.start())
# print('grammar.productions() =>')
# # Use string.replace(...) is to line-wrap the output.
# print(grammar.productions())

