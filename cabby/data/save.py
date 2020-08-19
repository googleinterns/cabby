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

'''Example command line method to extract Wikipedia and Wikidata items and save to file.
Example:
$ bazel-bin/cabby/data/save --region Pittsburgh --path geodata.txt
'''

from absl import app
from absl import flags
import json


from cabby.data import extract

FLAGS = flags.FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "region", None, ['Pittsburgh', 'Manhattan'],
    "Map areas: Manhattan or Pittsburgh.")
flags.DEFINE_string("path", None, "The path where the data will be saved.")


# Required flags.
flags.mark_flag_as_required("region")
flags.mark_flag_as_required("path")



def main(argv):
    del argv  # Unused.
    results = extract.get_data_by_region(FLAGS.region)
    print('The number of results items found is: {}'.format(
        len(results)))
    with open(FLAGS.path, 'w') as file:
        json.dump(results, file, sort_keys=True, indent=4)

if __name__ == '__main__':
    app.run(main)
