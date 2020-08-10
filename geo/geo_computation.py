'''Example command line method to compute the route between two given points.'''

from absl import app
from absl import flags

from walk import compute_route
from shapely.geometry.point import Point
import osmnx as ox

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

    # Compute graph over Manhattan.
    graph = ox.graph_from_place('Manhattan, New York City, New York, USA')

    # Convert a graph to nodes and edge GeoDataFrames.
    nodes, _ = ox.graph_to_gdfs(graph)

    print(
        compute_route(
            Point(FLAGS.orig_lon, FLAGS.orig_lat),
            Point(FLAGS.dest_lon, FLAGS.dest_lat), graph, nodes))


if __name__ == '__main__':
    app.run(main)
