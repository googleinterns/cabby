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

'''Example command line method to extract wikidata items.
Example:
$ bazel-bin/cabby/data/wikidata/extract_geofenced_wikidata_items \
  --region Pittsburgh
'''

from absl import app
from absl import flags

from cabby.data.wikidata import query
from cabby.geo import regions

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "region", None, regions.SUPPORTED_REGION_NAMES,
  regions.REGION_SUPPORT_MESSAGE)

# Required flags.
flags.mark_flag_as_required("region")

def main(argv):
  del argv  # Unused.
  results = query.get_geofenced_wikidata_items(FLAGS.region)
  print('The number of Wikidata items found is: {}'.format(len(results)))


if __name__ == '__main__':
  app.run(main)
