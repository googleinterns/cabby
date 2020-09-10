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

'''Example command line method to extract Wikipedia items.
Example:
$ bazel-bin/cabby/data/wikipedia/extract_wikipedia_items --titles="New_York_Stock_Exchange, Empire_State_Building"
'''


from absl import app
from absl import flags

from cabby.data.wikipedia import query

FLAGS = flags.FLAGS
flags.DEFINE_list(
  "titles", None,
  "List of Wikipedia titles to extract.")

# Required flags.
flags.mark_flag_as_required("titles")


def main(argv):
  del argv  # Unused.
  results = query.get_wikipedia_items(FLAGS.titles)
  for result in results:
    print(result)

  print('The number of Wikipedia items found is: {}'.format(
    len(results)))


if __name__ == '__main__':
  app.run(main)
