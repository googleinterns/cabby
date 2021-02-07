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

import geopandas as gpd
import pandas as pd
from typing import Any, Dict, Sequence
import os
from shapely.geometry import LineString, Polygon

from cabby.geo import util

import attr

_Geo_DataFrame_Driver = "GPKG"
VERSION = 0.4

@attr.s
class GeoEntity:
  """Construct a geo entity.

  `geo_landmarks` the geo landmarks (start and end points + pivots).
  `geo_features` the spatial features of the path.
  `route` the path from the start to end point.
  """

  geo_landmarks: Dict = attr.ib()
  geo_features: Dict = attr.ib()
  route: gpd.GeoDataFrame = attr.ib()

  @classmethod
  def add_entity(cls,
                 route: gpd.GeoDataFrame,
                 geo_landmarks: Dict[str, gpd.GeoDataFrame],
                 geo_features: Dict[str, Any]):
    geo_entity = GeoEntity({}, geo_features, route)
    for landmark_type, landmark in geo_landmarks.items():
      geo_landmark = GeoLandmark.add_pivot(landmark, landmark_type)
      geo_entity.geo_landmarks[geo_landmark.landmark_type] = geo_landmark
    return geo_entity

  @staticmethod
  def save(entities: Sequence, path_to_save: str):
    landmark_types = entities[0].geo_landmarks.keys()
    geo_types_all = {}
    empty_gdf = gpd.GeoDataFrame(
      columns=['osmid', 'geometry', 'main_tag'])
    for landmark_type in landmark_types:
      geo_types_all[landmark_type] = empty_gdf
    columns = ['geometry'] + list(entities[0].geo_features.keys())
    geo_types_all['path_features'] = gpd.GeoDataFrame(columns=columns)
    for entity in entities:
      for pivot_type, pivot in entity.geo_landmarks.items():
        geo_types_all[pivot_type] = geo_types_all[pivot_type].append(pivot.pivot_gdf)
      dict_features = {k: [v] for k,v in entity.geo_features.items()}
      pd_features = pd.DataFrame(dict_features)
      geometry = Polygon(LineString(entity.route['geometry'].tolist()))
      features_gdf = gpd.GeoDataFrame(pd_features, geometry=[geometry])
      geo_types_all['path_features'] = geo_types_all['path_features'].append(features_gdf)

    # Save pivots.
    if os.path.exists(path_to_save):
      mode = 'a'
    else:
      mode = 'w'
    for geo_type, pivots_gdf in geo_types_all.items():
      pivots_gdf.to_file(
          path_to_save, layer=geo_type, mode=mode, driver=_Geo_DataFrame_Driver)


  def rvs_sample(self, instructions: str, id: int):
    """Reformat a GeoEntity into an RVS sample."""
    geo_dict = {'instructions': instructions,
                'id': id,
                'version': VERSION}
    landmark_list = []
    for type_landmark, landmark in self.geo_landmarks.items():
      landmark_list.append(landmark.rvs_format())
    geo_dict['landmarks'] = landmark_list
    geo_dict['features'] = self.geo_features
    geo_dict['route_len'] = round(util.get_linestring_distance(self.route))
    return geo_dict


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
  geometry: Any = attr.ib()
  main_tag: str = attr.ib()
  pivot_gdf: gpd.GeoDataFrame = attr.ib()

  def __attrs_post_init__(self):
    columns_remove = self.pivot_gdf.keys().difference(['osmid', 'geometry', 'main_tag'])
    if len(columns_remove) == 0:
      return
    self.pivot_gdf.drop(columns_remove, inplace=True)

  def rvs_format(self):
    """Reformat a GeoLandmark into an RVS style."""

    return [self.landmark_type,
      self.osmid,
      self.main_tag]

  @classmethod
  def add_pivot(cls, pivot: gpd.GeoDataFrame, pivot_type: str):
    """Construct a GeoLandmark."""
    return GeoLandmark(
      pivot_type,
      pivot['osmid'],
      pivot['geometry'],
      pivot['main_tag'],
      pivot
    )





