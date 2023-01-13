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
"""The pre process of image."""

import numpy as np


def tile_images(img_nhwc, padding=5):
    """
    Tile N images into one big P*Q image

    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    Args:
        img_nhwc (np.ndarray): images to be tiled. The shape is (N, H, W, C).
    """
    # read the image as array
    img_nhwc = np.asarray(img_nhwc)
    # check the shape of the image
    n_images, height, width, n_channels = img_nhwc.shape
    # calculate the new height and width
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))

    # padding the images
    new_img_nhwc = np.zeros(
        (
            img_nhwc.shape[0],
            img_nhwc.shape[1] + 2 * padding,
            img_nhwc.shape[2] + 2 * padding,
            img_nhwc.shape[3],
        )
    )
    for i in range(img_nhwc.shape[0]):
        new_img_nhwc[i, :, :, :] = np.pad(
            img_nhwc[i, :, :, :],
            ((padding, padding), (padding, padding), (0, 0)),
            'constant',
            constant_values=0,
        )
    new_img_nhwc = np.array(
        list(new_img_nhwc) + [new_img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    height += 2 * padding
    width += 2 * padding

    # adjust the output shape
    out_image = new_img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image
