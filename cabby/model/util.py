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
'''Library to support models.'''

import scipy.stats

class DistanceProbability:
  '''Calculates Gamma distribution probability of a given distance in meters.

  For details of the Gamma distribution see:
    https://en.wikipedia.org/wiki/Gamma_distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

  With the default shape of 2, shorter distances receive lower probability and 
  then peaks at 1 (unscaled) before dropping again. This allows the probability
  to be used to prefer values close to some scaled mean (given via the 
  `scale_meters` parameter.) Changing the shape will change this interpretation
  but could be used to make the values more spread out (higher value of 
  `gamma_shape`) or start high and drop (`gamma_shape` < 1). In the later case,
  this could be useful for situations where being extremely close is preferable.
  
  `scale_meters`: A factor (in meters) that gives the overall scale in meters
    that distance probabilities are required for. If `gamma_shape` is 2, then
    the distribution reaches its peak probability around `scale_meters`.

  `gamma_shape`: The shape of the Gamma distribution.

  '''
  def __init__(self, scale_meters: float, gamma_shape=2):
    assert(scale_meters > 0.0)
    self.scale_meters = scale_meters
    self.gamma_dist = scipy.stats.gamma(gamma_shape)

  def __call__(self, dist_meters: float) -> float:
    '''Computes the probability for a given distance in meters.'''
    assert(dist_meters >= 0.0)
    return self.gamma_dist.pdf(dist_meters/self.scale_meters)

  def cdf(self, dist_meters: float) -> float:
    '''Computes the cumulative probability at a given distance in meters.'''
    assert(dist_meters >= 0.0)
    return self.gamma_dist.cdf(dist_meters/self.scale_meters)
