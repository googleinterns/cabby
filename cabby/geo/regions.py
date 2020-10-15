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
'''Defines supported regions and their corresponding polygons.'''

from typing import Dict


import attr
from shapely.geometry import box
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely import wkt

@attr.s
class Region:
  name: str = attr.ib()
  polygon: Polygon = attr.ib()

MANHATTAN_REGION = Region(
  name="Manhattan",
  polygon=wkt.loads('POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'),
)

PITTSBURGH_REGION = Region(
  name='Pittsburgh',
  polygon=box(miny=40.425, minx=-80.035, maxy=40.460, maxx=-79.930, ccw=True)
)

PITTSBURGH_SMALL_REGION = Region(
  name='Pittsburgh_small', 
  polygon=Point(-79.9837, 40.4273).buffer(0.0015)
)

DC_REGION = Region(
  name='DC', 
  polygon=box(minx=-77.02767, miny=38.96608, maxx=-77.02704, maxy=38.9686)
)

SUPPORTED_REGIONS = [
  MANHATTAN_REGION, 
  PITTSBURGH_REGION, 
  PITTSBURGH_SMALL_REGION, 
  DC_REGION
]

REGION_LOOKUP = dict(map(lambda r: (r.name, r), SUPPORTED_REGIONS))

SUPPORTED_REGION_NAMES = sorted(list(REGION_LOOKUP.keys()))
#ALLOWED_REGIONS = sorted(list(REGION_POLYGONS.keys()))
REGION_SUPPORT_MESSAGE = (
  'Supported regions: ' + ', '.join(SUPPORTED_REGION_NAMES))

def get_region(region_name: str) -> Polygon:
  try:
    return REGION_LOOKUP[region_name].polygon
  except:
    raise ValueError(
      f'Unsupported region {region_name}. '
      'Please choose one of ', SUPPORTED_REGION_NAMES)
