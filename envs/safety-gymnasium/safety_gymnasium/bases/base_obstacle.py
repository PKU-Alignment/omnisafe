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
"""Base class for obstacles."""

import abc
from dataclasses import dataclass

from safety_gymnasium.bases.base_agent import BaseAgent
from safety_gymnasium.utils.random_generator import RandomGenerator
from safety_gymnasium.world import Engine


@dataclass
class BaseObstacle(abc.ABC):
    """Base class for obstacles."""

    type: str = None
    name: str = None
    engine: Engine = None
    random_generator: RandomGenerator = None
    agent: BaseAgent = None

    def cal_cost(self):
        """Calculate the cost of the obstacle."""
        return {}

    def set_agent(self, agent):
        """Set the agent instance."""
        self.agent = agent
        self._specific_agent_config()

    def set_engine(self, engine: Engine):
        """Set the engine instance."""
        self.engine = engine

    def set_random_generator(self, random_generator):
        """Set the random generator instance."""
        self.random_generator = random_generator

    def process_config(self, config, layout, rots):
        """Process the config."""
        if hasattr(self, 'num'):
            assert (
                len(rots) == self.num
            ), 'The number of rotations should be equal to the number of obstacles.'
            for i in range(self.num):
                name = f'{self.name[:-1]}{i}'
                config[self.type][name] = self.get_config(xy_pos=layout[name], rot=rots[i])
                config[self.type][name].update({'name': name})
        else:
            assert len(rots) == 1, 'The number of rotations should be 1.'
            config[self.type][self.name] = self.get_config(xy_pos=layout[self.name], rot=rots[0])

    def _specific_agent_config(self):
        """Modify properties according to specific agent."""

    @property
    @abc.abstractmethod
    def pos(self):
        """Get the position of the obstacle."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_config(self, xy_pos, rot):
        """Get the config of the obstacle."""
        raise NotImplementedError


@dataclass
class Geom(BaseObstacle):
    """Base class for obstacles that are geoms."""

    type: str = 'geoms'


@dataclass
class FreeGeom(BaseObstacle):
    """Base class for obstacles that are free_geoms."""

    type: str = 'free_geoms'


@dataclass
class Mocap(BaseObstacle):
    """Base class for obstacles that are mocaps."""

    type: str = 'mocaps'

    def process_config(self, config, layout, rots):
        """Process the config."""
        if hasattr(self, 'num'):
            assert (
                len(rots) == self.num
            ), 'The number of rotations should be equal to the number of obstacles.'
            for i in range(self.num):
                mocap_name = f'{self.name[:-1]}{i}mocap'
                obj_name = f'{self.name[:-1]}{i}obj'
                layout_name = f'{self.name[:-1]}{i}'
                configs = self.get_config(xy_pos=layout[layout_name], rot=rots[i])
                config['free_geoms'][obj_name] = configs['obj']
                config['free_geoms'][obj_name].update({'name': obj_name})
                config['mocaps'][mocap_name] = configs['mocap']
                config['mocaps'][mocap_name].update({'name': mocap_name})
        else:
            assert len(rots) == 1, 'The number of rotations should be 1.'
            mocap_name = f'{self.name[:-1]}mocap'
            obj_name = f'{self.name[:-1]}obj'
            layout_name = self.name[:-1]
            configs = self.get_config(xy_pos=layout[layout_name], rot=rots[0])
            config['free_geoms'][obj_name] = configs['obj']
            config['free_geoms'][obj_name].update({'name': obj_name})
            config['mocaps'][mocap_name] = configs['mocap']
            config['mocaps'][mocap_name].update({'name': mocap_name})

    def set_mocap_pos(self, name, value):
        """Set the position of a mocap object."""
        body_id = self.engine.model.body(name).id
        mocap_id = self.engine.model.body_mocapid[body_id]
        self.engine.data.mocap_pos[mocap_id] = value

    @abc.abstractmethod
    def move(self):
        """Set mocap object positions before a physics step is executed."""
