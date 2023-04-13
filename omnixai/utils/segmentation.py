#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from PIL import Image as PilImage
from skimage.segmentation import quickshift, felzenszwalb, slic

from ..data.image import Image
from ..preprocessing.image import Resize


def image_segmentation(image, method="quickshift", **kwargs):
    if method == "quickshift":
        return quickshift(
            image,
            ratio=kwargs.get("ratio", 1.0),
            kernel_size=kwargs.get("kernel_size", 5),
            max_dist=kwargs.get("max_dist", 10),
            return_tree=kwargs.get("return_tree", False),
            sigma=kwargs.get("sigma", 0),
            convert2lab=kwargs.get("convert2lab", True),
            random_seed=kwargs.get("random_seed", 42),
            channel_axis=kwargs.get("channel_axis", -1),
        )
    elif method == "felzenszwalb":
        return felzenszwalb(
            image,
            scale=kwargs.get("scale", 1),
            sigma=kwargs.get("sigma", 0.8),
            min_size=kwargs.get("min_size", 20),
            multichannel=kwargs.get("multichannel", True),
            channel_axis=kwargs.get("channel_axis", -1),
        )
    elif method == "slic":
        return slic(
            image,
            n_segments=kwargs.get("n_segments", 100),
            compactness=kwargs.get("compactness", 10.0),
            max_num_iter=kwargs.get("max_num_iter", 10),
            sigma=kwargs.get("sigma", 0),
            spacing=kwargs.get("spacing", None),
            convert2lab=kwargs.get("convert2lab", None),
            enforce_connectivity=kwargs.get("enforce_connectivity", True),
            min_size_factor=kwargs.get("min_size_factor", 0.5),
            max_size_factor=kwargs.get("max_size_factor", 3),
            slic_zero=kwargs.get("slic_zero", False),
            start_label=kwargs.get("start_label", 1),
            mask=kwargs.get("mask", None),
            channel_axis=kwargs.get("channel_axis", -1),
        )
    elif method == "grid":
        return grid(image, min_size_per_edge=kwargs.get("min_size_per_edge", 10))
    else:
        raise NotImplementedError(f"Unknown image segmentation method: {method}")


def grid(image, min_size_per_edge=10):
    length = min(image.shape[:2])
    size = int(length / min_size_per_edge)
    if size == 0:
        raise ValueError("`min_size_per_edge` is larger than min(height, width).")

    a = int(np.ceil(image.shape[0] / size))
    b = int(np.ceil(image.shape[1] / size))
    mask = Image(data=np.arange(a * b).reshape((a, b)), batched=False)
    mask = Resize(size=(image.shape[0], image.shape[1]), resample=PilImage.NEAREST).transform(mask)
    return mask.to_numpy()[0]
