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
"""Example of plotting training curve."""


import argparse

from omnisafe.utils.plotter import Plotter


# For example, you can run the following command to plot the training curve:
# python plot.py --logdir omnisafe/examples/runs/PPOLag-{SafetyAntVelocity-v1}
# after training the policy with the following command:
# python train_policy.py --algo PPOLag --env-id SafetyAntVelocity-v1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='Steps')
    parser.add_argument('--value', '-y', default='Rewards', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--estimator', default='mean')
    args = parser.parse_args()

    plotter = Plotter()
    plotter.make_plots(
        args.logdir,
        args.legend,
        args.xaxis,
        args.value,
        args.count,
        smooth=args.smooth,
        select=args.select,
        exclude=args.exclude,
        estimator=args.estimator,
    )
