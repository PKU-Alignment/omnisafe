# Copyright 2023 OmniSafe Team. All Rights Reserved.
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

import os

from typer.testing import CliRunner

from omnisafe.utils.command_app import app


runner = CliRunner()
base_path = os.path.dirname(os.path.abspath(__file__))


def test_benchmark():
    result = runner.invoke(
        app,
        [
            'benchmark',
            'test_benchmark',
            '2',
            os.path.join(base_path, './saved_source/benchmark_config.yaml'),
        ],
    )
    assert result.exit_code == 0


def test_train():
    result = runner.invoke(
        app,
        [
            'train',
            '--algo',
            'PPO',
            '--total-steps',
            '1024',
            '--vector-env-nums',
            '1',
            '--custom-cfgs',
            'algo_cfgs:update_cycle',
            '--custom-cfgs',
            '512',
        ],
    )
    assert result.exit_code == 0


def test_train_config():
    result = runner.invoke(
        app, ['train-config', os.path.join(base_path, './saved_source/train_config.yaml')]
    )
    assert result.exit_code == 0


def test_eval():
    result = runner.invoke(
        app,
        [
            'eval',
            os.path.join(base_path, './saved_source/PPO-{SafetyPointGoal1-v0}'),
            '--num-episode',
            '1',
            '--width',
            '1',
            '--height',
            '1',
        ],
    )
    assert result.exit_code == 0
