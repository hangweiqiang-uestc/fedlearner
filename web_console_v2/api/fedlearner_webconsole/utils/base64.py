# Copyright 2021 The FedLearner Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
from base64 import b64encode, b64decode


def base64encode(s: str) -> str:
    return b64encode(s.encode('UTF-8')).decode('UTF-8')


def base64decode(s: str) -> str:
    return b64decode(s).decode('UTF-8')