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
"""Random generator."""

import numpy as np
from safety_gymnasium.utils.common_utils import ResamplingError


class RandomGenerator:
    """A random number generator that can be seeded and reset."""

    def __init__(self):
        """Initialize the random number generator."""
        self.random_generator = None
        self.placements = None
        self.placements_extents = None
        self.placements_margin = None
        self.layout = None

    def set_placements_info(self, placements, placements_extents, placements_margin):
        """Set the placements information."""
        self.placements = placements
        self.placements_extents = placements_extents
        self.placements_margin = placements_margin

    def set_random_seed(self, seed):
        """Instantiate a :class:`np.random.RandomState` object using given seed."""
        self.random_generator = np.random.RandomState(seed)  # pylint: disable=no-member

    def build_layout(self):
        """Rejection sample a placement of objects to find a layout."""
        for _ in range(10000):
            if self.sample_layout():
                return self.layout
        raise ResamplingError('Failed to sample layout of objects')

    def draw_placement(self, placements, keepout):
        """Sample an (x,y) location, based on potential placement areas.

        Summary of behavior:

        'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify
        rectangles in the XY-plane where an object could be placed.

        'keepout' describes how much space an object is required to have
        around it, where that keepout space overlaps with the placement rectangle.

        To sample an (x,y) pair, first randomly select which placement rectangle
        to sample from, where the probability of a rectangle is weighted by its
        area. If the rectangles are disjoint, there's an equal chance the (x,y)
        location will wind up anywhere in the placement space. If they overlap, then
        overlap areas are double-counted and will have higher density. This allows
        the user some flexibility in building placement distributions. Finally,
        randomly draw a uniform point within the selected rectangle.
        """
        if placements is None:
            choice = self.constrain_placement(self.placements_extents, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert constrained, 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.random_generator.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice
        return np.array(
            [self.random_generator.uniform(xmin, xmax), self.random_generator.uniform(ymin, ymax)]
        )

    def sample_layout(self):
        """Sample a single layout, returning True if successful, else False."""

        def placement_is_valid(xy, layout):  # pylint: disable=invalid-name
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(xy - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            return True

        layout = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                # pylint: disable-next=invalid-name
                xy = self.draw_placement(placements, keepout)
                if placement_is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        self.layout = layout
        return True

    def sample_goal_position(self):
        """Sample a new goal position and return True, else False if sample rejected."""
        placements, keepout = self.placements['goal']
        goal_xy = self.draw_placement(placements, keepout)
        for other_name, other_xy in self.layout.items():
            other_keepout = self.placements[other_name][1]
            dist = np.sqrt(np.sum(np.square(goal_xy - other_xy)))
            if dist < other_keepout + self.placements_margin + keepout:
                return False
        self.layout['goal'] = goal_xy
        return True

    def constrain_placement(self, placement, keepout):
        """Helper function to constrain a single placement by the keepout radius."""
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)

    def generate_rots(self, num=1):
        """Generate the rotations of the obstacle."""
        rots = [self.random_rot() for _ in range(num)]
        return rots

    def randn(self, *args, **kwargs):
        """Wrapper for :meth:`np.random.RandomState.randn`."""
        return self.random_generator.randn(*args, **kwargs)

    def binomial(self, *args, **kwargs):
        """Wrapper for :meth:`np.random.RandomState.binomial`."""
        return self.random_generator.binomial(*args, **kwargs)

    def random_rot(self):
        """Use internal random state to get a random rotation in radians."""
        return self.random_generator.uniform(0, 2 * np.pi)

    def choice(self, *args, **kwargs):
        """Wrapper for :meth:`np.random.RandomState.choice`."""
        return self.random_generator.choice(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        """Wrapper for :meth:`np.random.RandomState.uniform`."""
        return self.random_generator.uniform(*args, **kwargs)
