'''Example command line method to compute the route between two given points.'''

from absl import app
from absl import flags

from shapely.geometry.point import Point
import osmnx as ox

FLAGS = flags.FLAGS
flags.DEFINE_string("place", None, "map area - Manhattan\Pittsburgh.")

# Required flags.
flags.mark_flag_as_required("place")


def main(argv):
  del argv  # Unused.

  print(compute_route(Point(FLAGS.orig_lat,FLAGS.orig_lon), Point(FLAGS.dest_lat,FLAGS.dest_lon),graph,nodes))

if __name__ == '__main__':
  app.run(main)