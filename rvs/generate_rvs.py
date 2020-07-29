'''Example command line method to output simple RVS instructions.'''

from absl import app
from absl import flags

from speak import describe_route

FLAGS = flags.FLAGS
flags.DEFINE_string("ref_poi", None, "The reference POI.")
flags.DEFINE_string("goal_poi", None, "The goal POI.")

# Required flags.
flags.mark_flag_as_required("ref_poi")
flags.mark_flag_as_required("goal_poi")

def main(argv):
  del argv  # Unused.
  print(describe_route(FLAGS.ref_poi, FLAGS.goal_poi))

if __name__ == '__main__':
  app.run(main)