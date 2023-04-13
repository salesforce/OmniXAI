#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image as PilImage
from .base import ExplanationBase


def np_to_json(x):
    return json.dumps(x.tolist())


def json_to_np(s):
    return np.array(json.loads(s))


def pd_to_json(df):
    return json.dumps(json.loads(df.to_json(orient="index")))


def json_to_pd(s):
    return pd.read_json(s, orient="index")


class DefaultJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, pd.DataFrame) or isinstance(o, pd.Series):
            return o.to_dict()
        if isinstance(o, PilImage.Image):
            return np.array(o).tolist()
        if isinstance(o, ExplanationBase):
            return {
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "data": {k: deepcopy(v) for k, v in o.__dict__.items()}
            }
        return super().default(o)
