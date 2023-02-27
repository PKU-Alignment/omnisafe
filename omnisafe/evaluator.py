# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of Evaluator."""


class Evaluator:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms."""

    def __init__(self) -> None:
        pass

    def load_saved_model(self, save_dir: str, model_name: str) -> None:
        """Load saved model from save_dir.

        Args:
            save_dir (str): The directory of saved model.
            model_name (str): The name of saved model.

        """

    def load_running_model(self, env, actor) -> None:
        """Load running model from env and actor.

        Args:
            env (gym.Env): The environment.
            actor (omnisafe.actor.Actor): The actor.

        """

    def evaluate(self, num_episode: int, render: bool = False) -> None:
        """Evaluate the model.

        Args:
            num_episode (int): The number of episodes to evaluate.
            render (bool): Whether to render the environment.

        """
