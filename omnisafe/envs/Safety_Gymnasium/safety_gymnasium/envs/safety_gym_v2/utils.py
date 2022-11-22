import mujoco
import numpy as np


def update_dict_from(dict1, dict2):
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
    q = np.array(quat, dtype='float64')
    m = np.zeros(9, dtype='float64')
    mujoco.mju_quat2Mat(m, q)
    return m.reshape((3, 3))


def theta2vec(theta):
    """Convert an angle (in radians) to a unit vector in that angle around Z"""
    return np.array([np.cos(theta), np.sin(theta), 0.0])


def get_body_jacp(model, data, name, jacp=None):
    id = model.body(name).id
    if jacp is None:
        jacp = np.zeros(3 * model.nv).reshape(3, model.nv)
    jacp_view = jacp
    mujoco.mj_jacBody(model, data, jacp_view, None, id)
    return jacp


def get_body_xvelp(model, data, name):
    jacp = get_body_jacp(model, data, name).reshape((3, model.nv))
    xvelp = np.dot(jacp, data.qvel)
    return xvelp
