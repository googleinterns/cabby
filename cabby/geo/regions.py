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
  """Basic information defining a region on the earth's surface.
  
  `name`: An identifier used to select this region.
  `polygon`: A shapely Polygon defining the bounds of the region.
  `corner_sw`: A shapely Point corresponding to the south-west corner of the
    region's bounding box.
  `corner_ne`: A shapely Point corresponding to the north-east corner of the
    region's bounding box.
  """
  name: str = attr.ib()
  polygon: Polygon = attr.ib()
  corner_sw: Point = attr.ib()
  corner_ne: Point = attr.ib()

SUPPORTED_REGIONS = [
  Region(
    name='DC',
    polygon=box(minx=-77.02767, miny=38.96608, maxx=-77.02704, maxy=38.9686),
    corner_sw=Point(-77.02767, 38.96608),
    corner_ne=Point(-77.02704, 38.9686)
  ),
  Region(
    name="Manhattan",
    polygon=wkt.loads('POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'),
    corner_sw=Point(-74.0379, 40.6966),
    corner_ne=Point(-73.9293, 40.7963)
  ),
  Region(
    name='Pittsburgh',
    polygon=box(minx=-80.035, miny=40.425, maxx=-79.930, maxy=40.460, ccw=True),
    corner_sw=Point(-80.035, 40.425),
    corner_ne=Point(-79.930, 40.460)
  ),
  Region(
    name='Pittsburgh_small',
    polygon=Point(-79.9837, 40.4273).buffer(0.0015),
    corner_sw=Point(-79.9837, 40.4273),
    corner_ne=Point(-79.953274, 40.444565)
  ),
  Region(
    name='UTAustin',
    polygon=box(minx=-97.7418, miny=30.2836, maxx=-97.7360, maxy=30.2872),
    corner_sw=Point(-97.7418, 30.2836),
    corner_ne=Point(-97.7360, 30.2872)
  )  
]

REGION_LOOKUP = dict(map(lambda r: (r.name, r), SUPPORTED_REGIONS))

SUPPORTED_REGION_NAMES = sorted(list(REGION_LOOKUP.keys()))

REGION_SUPPORT_MESSAGE = (
  'Supported regions: ' + ', '.join(SUPPORTED_REGION_NAMES))

def get_region(region_name: str) -> Region:
  try:
    return REGION_LOOKUP[region_name]
  except:
    raise ValueError(
      f'Unsupported region {region_name}. '
      'Please choose one of ', SUPPORTED_REGION_NAMES)
