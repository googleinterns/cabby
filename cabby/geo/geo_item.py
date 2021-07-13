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
'''Basic classes and functions for RVSPath items.'''

from absl import logging
import geopandas as gpd
import pandas as pd
from typing import Any, Dict, Sequence, Tuple, List
import os
from shapely.geometry.base import BaseGeometry
from shapely.geometry.point import Point
from shapely.geometry import LineString, Polygon

from cabby.geo import util

import attr

_Geo_DataFrame_Driver = "GPKG"
VERSION = 0.5

@attr.s
class GeoEntity:
  """Construct a geo entity.

  `geo_landmarks` the geo landmarks (start and end points and pivots).
  `geo_features` the spatial features of the path.
  Dictionary values can be of either type str or int.
  `route` the path from the start to end point.
  `states` a sequence of states (Point, bearing, and visual description).
  """

  geo_landmarks: Dict[str, gpd.GeoDataFrame] = attr.ib()
  geo_features: Dict[str, Any] = attr.ib()
  route: gpd.GeoDataFrame = attr.ib()
  states: List[Tuple[Point,float, str]] = attr.ib()

  @classmethod
  def add_entity(cls,
                 route: gpd.GeoDataFrame,
                 geo_landmarks: Dict[str, gpd.GeoDataFrame],
                 geo_features: Dict[str, Any],
                 states: List[Tuple[Point, float, str]]):
    geo_entity = GeoEntity({}, geo_features, route, states)
    for landmark_type, landmark in geo_landmarks.items():
      geo_landmark = GeoLandmark.create_from_pivot(landmark, landmark_type)
      geo_entity.geo_landmarks[geo_landmark.landmark_type] = geo_landmark
    return geo_entity


@attr.s
class GeoLandmark:
  """Construct a geo landmark.

  `landmark_type` type of landmark.
  `osmid` osmid of the landmark.
  `geometry` the geometry of the landmark.
  `main_tag` the tag of the entity that will appear in the instruction.
  `pivot_gdf` the GeoDataFrame format of the entity.
  """
  landmark_type: str = attr.ib()
  osmid: int = attr.ib()
  geometry: BaseGeometry = attr.ib()
  main_tag: str = attr.ib()
  pivot_gdf: gpd.GeoDataFrame = attr.ib()

  def __attrs_post_init__(self):
    columns_remove = self.pivot_gdf.keys().difference(['osmid', 'geometry', 'main_tag'])
    if len(columns_remove) > 0:
      self.pivot_gdf.drop(columns_remove, inplace=True)

  def to_rvs_format(self):
    """Reformat a GeoLandmark into an RVS style."""
    centroid = util.tuple_from_point(
      self.geometry.centroid) if self.geometry else None

    return [
      self.landmark_type,
      self.osmid,
      self.main_tag,
      centroid
    ]

  @classmethod
  def create_from_pivot(cls, pivot: gpd.GeoDataFrame, pivot_type: str):
    """Construct a GeoLandmark."""
    return GeoLandmark(
      pivot_type,
      pivot['osmid'],
      pivot['geometry'],
      pivot['main_tag'],
      pivot
    )


@attr.s
class RVSSample:
  """Construct a RVS sample.

  `geo_landmarks` the geo landmarks (start and end points + pivots).
  `geo_features` the spatial features of the path.
  Dictionary values can be of either type str or int.
  `route_len` the length of the path from the start to end point.
  `instructions` the text instructing how to get from start location to goal location.
  `id` the sample id.
  `version` the datasets version.
  `entity_span` the entity span in the instruction.
  Key: entity name. Value: tuple of start and end of the span.
  `states` a list of states.
  """

  geo_landmarks: Dict[str, gpd.GeoDataFrame] = attr.ib()
  geo_features: Dict[str, Any] = attr.ib()
  route_len: int = attr.ib()
  instructions: str = attr.ib()
  id: int = attr.ib()
  version: float = attr.ib()
  entity_span: Dict[str, Tuple[int, int]] = attr.ib()
  states: str = attr.ib()

  @classmethod
  def to_rvs_sample(self,
                    instructions: str,
                    id: int,
                    geo_entity: GeoEntity,
                    entity_span: Dict[str, Tuple[int, int]],
):
    """Construct a RVS sample from GeoEntity."""
    landmark_list = {}
    for type_landmark, landmark in geo_entity.geo_landmarks.items():
      landmark_list[type_landmark]= landmark.to_rvs_format()
    route_length = round(util.get_linestring_distance(geo_entity.route))
    positions = list(geo_entity.states['geometry'].coords)
    bearings = geo_entity.states['angle']
    descriptions = geo_entity.states['descriptions']

    states = [(util.tuple_from_point(p), b, d) for p, b, d in zip(positions, bearings, descriptions)]

    return RVSSample(
              landmark_list,
              geo_entity.geo_features,
              route_length,
              instructions,
              id,
              VERSION,
              entity_span,
              states
)

def save(entities: Sequence[GeoEntity], path_to_save: str):
  path_to_save = os.path.abspath(path_to_save)

  landmark_types = entities[0].geo_landmarks.keys()
  geo_types_all = {}
  empty_gdf = gpd.GeoDataFrame(
    columns=['osmid', 'geometry', 'main_tag'])
  for landmark_type in landmark_types:
    geo_types_all[landmark_type] = empty_gdf
  columns = ['geometry'] + list(entities[0].geo_features.keys())
  geo_types_all['path_features'] = gpd.GeoDataFrame(columns=columns)
  geo_types_all['states'] = gpd.GeoDataFrame(columns=['geometry', 'angle', 'descriptions'])

  for entity in entities:
    for pivot_type, pivot in entity.geo_landmarks.items():
      geo_types_all[pivot_type] = geo_types_all[pivot_type].append(pivot.pivot_gdf)
    dict_features = {k: [v] for k,v in entity.geo_features.items()}
    pd_features = pd.DataFrame(dict_features)
    geometry = LineString(entity.route['geometry'].tolist())
    features_gdf = gpd.GeoDataFrame(pd_features, geometry=[geometry])
    geo_types_all['path_features'] = geo_types_all['path_features'].append(features_gdf)

    geometry = LineString([e[0] for e in entity.states])
    angle_str = ','.join([str(e[1]) for e in entity.states])
    desc_str = ';'.join([e[2] for e in entity.states])
    pd_angles_desc = pd.DataFrame({'angle': [angle_str], 'descriptions': [desc_str]})
    states_gdf = gpd.GeoDataFrame(pd_angles_desc, geometry=[geometry])

    geo_types_all['states'] = geo_types_all['states'].append(states_gdf)

  # Save pivots.
  if os.path.exists(path_to_save):
    mode = 'a'
  else:
    mode = 'w'
  for geo_type, pivots_gdf in geo_types_all.items():
    pivots_gdf.to_file(
        path_to_save, layer=geo_type, mode=mode, driver=_Geo_DataFrame_Driver)

  logging.info(f"Saved {len(entities)} entities to => {path_to_save}")


