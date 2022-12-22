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
"""Early terminated wrappe"""

import torch

from omnisafe.wrappers.on_policy_wrapper import OnPolicyEnvWrapper
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


@WRAPPER_REGISTRY.register
class EarlyTerminatedEnvWrapper(OnPolicyEnvWrapper):  # pylint: disable=too-many-instance-attributes
    """EarlyTerminatedEnvWrapper."""

    # pylint: disable-next=too-many-locals
    def roll_out(self, agent, buf, logger):
        """Collect data and store to experience buffer.
        Terminated when the episode is done or the episode length is larger than max_ep_len
        or cost is unequal to 0."""
        obs, _ = self.env.reset()
        ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
        for step_i in range(self.local_steps_per_epoch):
            action, value, cost_value, logp = agent.step(torch.as_tensor(obs, dtype=torch.float32))
            next_obs, reward, cost, done, truncated, _ = self.step(action)
            ep_ret += reward
            ep_costs += (self.cost_gamma**ep_len) * cost
            ep_len += 1

            # Save and log
            # Notes:
            #   - raw observations are stored to buffer (later transformed)
            #   - reward scaling is performed in buffer
            buf.store(
                obs=obs,
                act=action,
                rew=reward,
                val=value,
                logp=logp,
                cost=cost,
                cost_val=cost_value,
            )

            # Store values for statistic purpose
            if self.use_cost:
                logger.store(**{'Values/V': value, 'Values/C': cost_value})
            else:
                logger.store(**{'Values/V': value})

            # Update observation
            obs = next_obs

            timeout = ep_len == self.max_ep_len
            terminal = done or timeout or truncated or cost
            epoch_ended = step_i == self.local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, value, cost_value, _ = agent(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    value, cost_value = 0.0, 0.0

                # Automatically compute GAE in buffer
                buf.finish_path(value, cost_value, penalty_param=float(self.penalty_param))

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    logger.store(
                        **{
                            'Metrics/EpRet': ep_ret,
                            'Metrics/EpLen': ep_len,
                            'Metrics/EpCost': ep_costs,
                        }
                    )
                ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
                obs, _ = self.env.reset()
