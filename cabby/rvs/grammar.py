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


MAIN_PART_NO_GOAL = ["X INTERSECTIONS CARDINAL_DIRECTION","NUMBER_INTERSECTIONS intersections past MAIN_PIVOT. ",
"to the NEXT block intersection past MAIN_PIVOT",
"When you pass MAIN_PIVOT, you'll be just NUMBER_INTERSECTIONS intersections away",
"through PARK to MAIN_PIVOT",
"past MAIN_PIVOT"
]

V1 = ['Go', 'Walk', 'Come', 'Head' ]
V2 = ['Meet']
GOAL_END = ['and meet at GOAL']
NEAR_GOAL = ["It(if goal in beging) GOAL will be near NEAR_PIVOT."]


prods = [Production(Nonterminal('S'), (Nonterminal('V1'), Nonterminal('M'))), 

Production(Nonterminal('M'), (Nonterminal('MAIN_PART_NO_GOAL'), Nonterminal('G'))),
Production(Nonterminal('G'), (Nonterminal('GOAL_END'), Nonterminal('F'))),
Production(Nonterminal('F'), (Nonterminal('NEAR_GOAL'), Nonterminal('ADDITION'))),

Production(Nonterminal('F'), (Nonterminal('NEAR_GOAL'), Nonterminal('E'))),
         ]

for phrase in NEAR_GOAL:
  rule = Production(Nonterminal('NEAR_GOAL'), (phrase,))
  prods.append(rule)

for phrase in GOAL_END:
  rule = Production(Nonterminal('GOAL_END'), (phrase,))
  prods.append(rule)

for phrase in MAIN_PART_NO_GOAL:
  rule = Production(Nonterminal('MAIN_PART_NO_GOAL'), (phrase,))
  prods.append(rule)

for phrase in V1:
  rule = Production(Nonterminal('V1'), (phrase,))
  prods.append(rule)

rule = Production(Nonterminal('E'), ('.',))
prods.append(rule)


grammar = CFG(Nonterminal('S'), prods)
# grammar = CFG.fromstring("""
#   S -> V1 MAIN_PART_NO_GOAL
#   MAIN_PART_NO_GOAL -> 'X INTERSECTIONS CARDINAL_DIRECTION' | 'NUMBER_INTERSECTIONS intersections past MAIN_PIVOT.'
#   V1 -> 'Go' | 'Walk' | 'Come' | 'Head' 
#   V2 -> 'Meet'
# """)

for sentence in generate(grammar, n=50):
     print(' '.join(sentence))
# print('A Grammar:', grammar)
# print('grammar.start()   =>', grammar.start())
# print('grammar.productions() =>')
# # Use string.replace(...) is to line-wrap the output.
# print(grammar.productions())

