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
"""Training algorithms with OmniSafe."""

import argparse
import typer

import omnisafe

exp = typer.Typer()


@exp.command()
def train(
    algo: str = typer.Option(..., '--algo', help='Algorithm name'),
    env_id: str = typer.Option(..., '--env-id', help='Environment name'),
    parallel: int = 1,
    total_steps: int = 1638400,
    device: str = 'cpu',
    vector_env_nums: int = 16,
    torch_threads: int = 16,
):
    agent = omnisafe.Agent(
        algo,
        env_id,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )
    agent.learn()


if __name__ == '__main__':
    typer.run(train)
