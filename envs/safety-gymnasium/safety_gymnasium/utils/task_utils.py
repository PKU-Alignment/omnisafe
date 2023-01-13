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
"""Utils for task classes."""

import re

import mujoco
import numpy as np


def get_task_class_name(task_id):
    """Help to translate task_id into task_class_name."""
    class_name = re.findall('[A-Z][^A-Z]*', task_id.split('-')[0])[-1]
    class_name = class_name[:-1] + 'Level' + class_name[-1]
    return class_name


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
    """Get specific body's Cartesian velocity."""
    jacp = get_body_jacp(model, data, name).reshape((3, model.nv))
    xvelp = np.dot(jacp, data.qvel)
    return xvelp


def add_velocity_marker(viewer, pos, vel, cost, velocity_threshold):
    """Add a marker to the viewer to indicate the velocity of the agent."""
    pos = pos + np.array([0, 0, 0.6])
    safe_color = np.array([0.2, 0.8, 0.2, 0.5])
    unsafe_color = np.array([0.5, 0, 0, 0.5])

    if cost:
        color = unsafe_color
    else:
        vel_ratio = vel / velocity_threshold
        color = safe_color * (1 - vel_ratio)

    viewer.add_marker(
        pos=pos,
        size=0.2 * np.ones(3),
        type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
        rgba=color,
        label='',
    )


def clear_viewer(viewer):
    """Clear the viewer's all markers and overlays."""
    # pylint: disable=protected-access
    viewer._markers[:] = []
    viewer._overlays.clear()
