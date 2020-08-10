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

'''Example command line method to output simple RVS instructions.'''

from absl import app
from absl import flags

from rvs import speak

FLAGS = flags.FLAGS
flags.DEFINE_string("ref_poi", None, "The reference POI.")
flags.DEFINE_string("goal_poi", None, "The goal POI.")

# Required flags.
flags.mark_flag_as_required("ref_poi")
flags.mark_flag_as_required("goal_poi")


def main(argv):
    del argv  # Unused.
    print(speak.describe_route(FLAGS.ref_poi, FLAGS.goal_poi))


if __name__ == '__main__':
    app.run(main)
