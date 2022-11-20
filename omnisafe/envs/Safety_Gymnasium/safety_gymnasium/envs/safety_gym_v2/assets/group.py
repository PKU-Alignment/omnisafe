"""Group"""

GROUP = {
    # Groups are a mujoco-specific mechanism for selecting which geom objects to "see"
    # We use these for raycasting lidar, where there are different lidar types.
    # These work by turning "on" the group to see and "off" all the other groups.
    # See obs_lidar_natural() for more.
    'goal': 0,
    'box': 1,
    'button': 1,
    'wall': 2,
    'pillar': 2,
    'hazard': 3,
    'vase': 4,
    'gremlin': 5,
    'circle': 6,
    'apple': 7,
    'orange': 8,
}
