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

import argparse
import time

import torch

import omnisafe


torch.set_num_threads(5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        default='SafeLoop',
        help='Choose from: {MBPPOLag, SafeLoop',
    )
    parser.add_argument(
        '--env-id',
        type=str,
        default='SafetyPointGoal1-v0',
        help='MBPPOLag only support SafetyPointGoal3-v0, SafetyCarGoal1-v0,\
        and SafeLoop only support SafetyPointGoal1-v0, SafetyCarGoal3-v0',
    )
    parser.add_argument(
        '--parallel', default=1, type=int, help='Number of paralleled progress for calculations.'
    )
    args, unparsed_args = parser.parse_known_args()

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [eval(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    env = omnisafe.EnvModelBased(args.algo, args.env_id)
    agent = omnisafe.Agent(args.algo, env, parallel=args.parallel, custom_cfgs=unparsed_dict)
    agent.learn()
