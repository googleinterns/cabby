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
from typing import Any, Dict, Sequence, Tuple
import os
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, Polygon

from cabby.geo import util

import attr

_Geo_DataFrame_Driver = "GPKG"
VERSION = 0.5
NOT_PRIVIEW_TAGS = [
  'osmid', 'main_tag','unique_id',"element_type", "node_ele",
  "gnis:Class", "gnis:County", "York_gnis", "alpha", "import_uuid", "hours", "gnis:ST_num",
  "gnis:id", "in", "nycdoitt:bin", "gnis:feature_id", "element_type", "phone",
  "website", "addr:housenumber", "contact:facebook", "contact:instagram", "opening_hours",
  "reservation", "wikidata"] 


@attr.s
class GeoEntity:
  """Construct a geo entity.

  `geo_landmarks` the geo landmarks (start and end points + pivots).
  `geo_features` the spatial features of the path.
  Dictionary values can be of either type str or int.
  `route` the path from the start to end point.
  """

  geo_landmarks: Dict[str, gpd.GeoDataFrame] = attr.ib()
  geo_features: Dict[str, Any] = attr.ib()
  route: gpd.GeoDataFrame = attr.ib()

  @classmethod
  def add_entity(cls,
                 route: gpd.GeoDataFrame,
                 geo_landmarks: Dict[str, gpd.GeoDataFrame],
                 geo_features: Dict[str, Any]):
    geo_entity = GeoEntity({}, geo_features, route)
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
    landmark_dict = {}
    for k,v in self.pivot_gdf.to_dict().items():
      if str(v)=='nan':
        continue
      if isinstance(v,str) and v and k not in NOT_PRIVIEW_TAGS:
        if  not(
          self.landmark_type in ['end_point', 'near_pivot', "beyond_pivot"] and 'name' in k):
          landmark_dict[k.replace('_', ' ')] = v.replace('_', ' ')
    
    landmark_desc_list = [
      f"{t}: {v}" for t, v in landmark_dict.items()]

    if len(landmark_desc_list)>0:
      landmark_desc_list.insert(0, "________________________")

    landmark_desc_list.insert(0, self.main_tag.replace("_", " "))
    self.pivot_gdf['pivot_view'] = ';'.join(landmark_desc_list)
    self.pivot_gdf_all = self.pivot_gdf
    columns_remove = self.pivot_gdf.keys().difference(
      ['osmid', 'geometry', 'main_tag', 'pivot_view'])
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
      str(pivot['main_tag']),
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
  """

  geo_landmarks: Dict[str, gpd.GeoDataFrame] = attr.ib()
  geo_features: Dict[str, Any] = attr.ib()
  route_len: int = attr.ib()
  instructions: str = attr.ib()
  id: int = attr.ib()
  version: float = attr.ib()
  entity_span: Dict[str, Tuple[int, int]] = attr.ib()

  @classmethod
  def to_rvs_sample(self,
                    instructions: str,
                    id: int,
                    geo_entity: GeoEntity,
                    entity_span: Dict[str, Tuple[int, int]]):
    """Construct a RVS sample from GeoEntity."""
    landmark_list = {}
    for type_landmark, landmark in geo_entity.geo_landmarks.items():
      landmark_list[type_landmark]= landmark.to_rvs_format()
    route_length = round(util.get_linestring_distance(geo_entity.route))
    return RVSSample(
              landmark_list,
              geo_entity.geo_features,
              route_length,
              instructions,
              id,
              VERSION,
              entity_span)



def save(entities: Sequence[GeoEntity], path_to_save: str):
  path_to_save = os.path.abspath(path_to_save)

  landmark_types = entities[0].geo_landmarks.keys()
  geo_types_all = {}
  empty_gdf = gpd.GeoDataFrame(
    columns=['osmid', 'geometry', 'main_tag', 'pivot_view'])
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

  logging.info(f"Saved {len(entities)} entities to => {path_to_save}")





