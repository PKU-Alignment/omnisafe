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
"""Keyboard viewer."""

import glfw
import imageio
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import Viewer


class KeyboardViewer(Viewer):  # pylint: disable=too-many-instance-attributes
    """Keyboard viewer."""

    def __init__(self, model, data, custom_key_press_callback):
        super().__init__(model, data)
        self._custom_key_press_callback = custom_key_press_callback

    # pylint: disable-next=too-many-arguments, too-many-branches
    def _key_callback(self, window, key, scancode, action, mods):
        """Callback for keyboard events."""
        if action != glfw.RELEASE:
            pass
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # pylint: disable=no-member
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # pylint: disable=no-member
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (
                    glfw.get_framebuffer_size(self.window)[1],
                    glfw.get_framebuffer_size(self.window)[0],
                    3,
                ),
                dtype=np.uint8,
            )
            mujoco.mjr_readPixels(img, None, self.viewport, self.con)  # pylint: disable=no-member
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            # pylint: disable=no-member
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
            # pylint: enable=no-member
        # Display coordinate frames
        elif key == glfw.KEY_E:
            self.vopt.frame = 1 - self.vopt.frame
        # Hide overlay menu
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        if key in (glfw.KEY_I, glfw.KEY_J, glfw.KEY_K, glfw.KEY_L):
            self._custom_key_press_callback(key=key, action=action)

        # Quit
        if key == glfw.KEY_ESCAPE:
            print('Pressed ESC')
            print('Quitting.')
            glfw.destroy_window(self.window)
            glfw.terminate()
