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
"""One example for evaluate saved policy."""

import os

import omnisafe


# Just fill your experiment's log directory in here.
# Such as: ~/omnisafe/runs/SafetyPointGoal1-v0/CPO/seed-000-2022-12-25_14-45-05
LOG_DIR = ''

if __name__ == '__main__':
    evaluator = omnisafe.Evaluator()
    for item in os.scandir(os.path.join(LOG_DIR, 'torch_save')):
        if item.is_file() and item.name.split('.')[-1] == 'pt':
            evaluator.load_saved_model(save_dir=LOG_DIR, model_name=item.name)
            evaluator.render(num_episode=10, camera_name='track', width=256, height=256)
