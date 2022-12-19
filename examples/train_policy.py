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
"""Example of training a policy with OmniSafe."""

import argparse

import omnisafe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        default='PPOSimmerQ',
        help='Choose from: {PolicyGradient, PPO, PPOLag, NaturalPG,'
        ' TRPO, TRPOLag, PDO, NPGLag, CPO, PCPO, FOCOPS, CPPOPid, CUP, PPOSaute,'
        'PPOSimmerPID, PPOSimmerQ, PPOEarlyTerminated',
    )
    parser.add_argument(
        '--env-id',
        type=str,
        default='SafetyPointGoal1-v0',
        help='The name of test environment',
    )
    parser.add_argument(
        '--parallel', default=1, type=int, help='Number of paralleled progress for calculations.'
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_dict = dict(zip(keys, values))
    # env = omnisafe.Env(args.env_id)
    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        parallel=args.parallel,
        custom_cfgs=unparsed_dict,
    )
    agent.learn()
