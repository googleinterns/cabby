'''Example command line method to output simple GEO route computation.'''

from absl import app
from absl import flags

from walk import compute_route

FLAGS = flags.FLAGS
flags.DEFINE_float("orig_lat", None, "origin latitude.")
flags.DEFINE_float("orig_lon", None, "origin longtitude.")
flags.DEFINE_float("dest_lat", None, "destination latitude.")
flags.DEFINE_float("dest_lon", None, "destination longtitude.")

# Required flags.
flags.mark_flag_as_required("orig_lat")
flags.mark_flag_as_required("orig_lon")
flags.mark_flag_as_required("dest_lat")
flags.mark_flag_as_required("dest_lon")

def main(argv):
  del argv  # Unused.
  print(compute_route((FLAGS.orig_lat,FLAGS.orig_lon), (FLAGS.dest_lat,FLAGS.dest_lon)))

if __name__ == '__main__':
  app.run(main)