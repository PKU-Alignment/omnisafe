"""common"""
import numpy as np


def quat2zalign(quat):
    """From quaternion, extract z_{ground} dot z_{body}"""
    # z_{body} from quaternion [a,b,c,d] in ground frame is:
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z_{ground} = [0,0,1] is
    # z_{body} dot z_{ground} = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat
    return a**2 - b**2 - c**2 + d**2


def convert(v):
    """Convert a value into a string for mujoco XML"""
    if isinstance(v, (int, float, str)):
        return str(v)
    # Numpy arrays and lists
    return ' '.join(str(i) for i in np.asarray(v))


def rot2quat(theta):
    """Get a quaternion rotated only about the Z axis"""
    return np.array([np.cos(theta / 2), 0, 0, np.sin(theta / 2)], dtype='float64')


# pylint: disable=W0107
class ResamplingError(AssertionError):
    """Raised when we fail to sample a valid distribution of objects or goals"""

    pass


class MujocoException(Exception):
    pass
