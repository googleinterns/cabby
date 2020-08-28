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

'''Function to create logger.'''

import logging
from logging.handlers import RotatingFileHandler


def create_logger(save_path, loggername):
    logger = logging.getLogger(loggername)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path, 'a', 1000000, 1)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.INFO)
    logger.addHandler(steam_handler)

    return logger
