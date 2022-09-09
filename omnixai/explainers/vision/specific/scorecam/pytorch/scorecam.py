#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
from typing import Callable
from abc import abstractmethod

from omnixai.utils.misc import AutodocABCMeta
from omnixai.data.image import Image
from omnixai.preprocessing.image import Resize
from omnixai.utils.misc import is_torch_available
from omnixai.explanations.image.pixel_importance import PixelImportance

if not is_torch_available():
    raise EnvironmentError("Torch cannot be found.")
else:
    import torch
    import torch.nn as nn
