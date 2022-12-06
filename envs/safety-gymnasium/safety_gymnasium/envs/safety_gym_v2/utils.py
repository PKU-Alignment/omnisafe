# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
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
# ==============================================================================
"""utils"""

import mujoco
import numpy as np


def update_dict_from(dict1, dict2):
    """Inplace and Recursively(depth1) update dict1 via dict2.

    If they have the same key which has :class:`dict` type value.
    If they have the same key, but the value in dict1 is not :class:`dict` type,
    then replace that value with value in dict2.
    """
    # pylint: disable=invalid-name
    for k1, v1 in dict1.items():
        if k1 in dict2:
            if isinstance(v1, dict):
                v1.update(dict2[k1])
            else:
                v1 = dict2[k1]

    for k2, v2 in dict2.items():
        if k2 not in dict1:
            dict1[k2] = v2


def quat2mat(quat):
    """Convert Quaternion to a 3x3 Rotation Matrix using mujoco"""
    # pylint: disable=invalid-name
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco.mju_quat2Mat(m, q)  # pylint: disable=no-member
    return m.reshape((3, 3))


def theta2vec(theta):
    """Convert an angle (in radians) to a unit vector in that angle around Z"""
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def get_body_jacp(model, data, name, jacp=None):
    """Get specific body's Jacobian via mujoco."""
    id = model.body(name).id  # pylint: disable=redefined-builtin, invalid-name
    if jacp is None:
        jacp = np.zeros(3 * model.nv).reshape(3, model.nv)
    jacp_view = jacp
    mujoco.mj_jacBody(model, data, jacp_view, None, id)  # pylint: disable=no-member
    return jacp


def get_body_xvelp(model, data, name):
    """Get specific body's cartesian velocity."""
    jacp = get_body_jacp(model, data, name).reshape((3, model.nv))
    xvelp = np.dot(jacp, data.qvel)
    return xvelp
