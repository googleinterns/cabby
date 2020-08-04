from absl import app
from absl import flags

from shapely.geometry.point import Point
import osmnx as ox
from map_structure import Map

FLAGS = flags.FLAGS
flags.DEFINE_string("place", None, "map area - Manhattan or Pittsburgh.")

# Required flags.
flags.mark_flag_as_required("place")


def main(argv):
    del argv  # Unused.

    pittsburgh_map = Map(FLAGS.place)


if __name__ == '__main__':
    app.run(main)
