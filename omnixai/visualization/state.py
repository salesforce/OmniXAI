from collections import OrderedDict
from ..data.tabular import Tabular


class State:
    views = ["local", "global", "prediction", "data"]

    def __init__(self):
        self.class_names = None
        self.params = None
        self.instances = None
        self.instance_indices = []

        self.explanations = {
            view: {} for view in State.views
        }
        self.state_params = {
            "num_figures_per_row": {},
            "plots": {},
            "display_plots": {},
            "display_instance": {}
        }

    def set(
            self,
            instances,
            local_explanations,
            global_explanations,
            data_explanations,
            prediction_explanations,
            class_names,
            params
    ):
        self.class_names = class_names
        self.params = {} if params is None else params
        self.instances = instances
        self.instance_indices = list(range(self.instances.num_samples())) \
            if instances is not None else []

        self.set_explanations("local", local_explanations)
        self.set_explanations("global", global_explanations)
        self.set_explanations("data", data_explanations)
        self.set_explanations("prediction", prediction_explanations)

        for view, explanations in self.explanations.items():
            self.set_plots(view, [name for name in explanations.keys()])
            self.set_display_plots(view, self.get_plots(view))
            self.set_display_instance(view, 0)
            self.set_num_figures_per_row(view, 2)

    def set_explanations(self, view, explanations):
        assert view in self.explanations
        if explanations is not None:
            self.explanations[view] = explanations

    def get_explanations(self, view):
        return self.explanations[view]

    def has_explanations(self):
        for explanations in self.explanations.values():
            if len(explanations) > 0:
                return True
        return False

    def set_num_figures_per_row(self, view, n):
        self.state_params["num_figures_per_row"][view] = n

    def get_num_figures_per_row(self, view):
        return self.state_params["num_figures_per_row"][view]

    def set_plots(self, view, plots):
        self.state_params["plots"][view] = plots

    def get_plots(self, view):
        return self.state_params["plots"][view]

    def set_display_plots(self, view, plots):
        self.state_params["display_plots"][view] = plots

    def get_display_plots(self, view):
        return self.state_params["display_plots"][view]

    def set_display_instance(self, view, index):
        self.state_params["display_instance"][view] = index

    def get_display_instance(self, view):
        return self.state_params["display_instance"][view]

    def set_param(self, view, param, value):
        self.state_params[param][view] = value

    def get_param(self, view, param):
        return self.state_params[param][view]

    def is_tabular(self):
        return isinstance(self.instances, Tabular)


class WhatifState:

    def __init__(self):
        self.class_names = None
        self.params = None
        self.instances = None
        self.instance_indices = []
        self.explainer = None
        self.features = None

        self.state_params = {
            "display_plots": [],
            "display_instance": 0,
            "instances-a": None,
            "instances-b": None,
            "what-if-a": {},
            "what-if-b": {}
        }

    def set(
            self,
            instances,
            local_explanations,
            class_names,
            params,
            explainer
    ):
        self.class_names = class_names
        self.params = {} if params is None else params
        self.instances = instances
        self.instance_indices = list(range(self.instances.num_samples())) \
            if instances is not None else []
        self.explainer = explainer
        self.features = self._extract_feature_values()

        self.state_params["display_plots"] = [name for name in local_explanations.keys()]
        self.state_params["instances-a"] = instances.copy()
        self.state_params["instances-b"] = instances.copy()
        self.state_params["what-if-a"] = local_explanations
        self.state_params["what-if-b"] = local_explanations

    def set_explanations(self, view, explanations):
        assert view in ["what-if-a", "what-if-b"]
        if explanations is not None:
            self.state_params[view] = explanations

    def get_explanations(self, view):
        assert view in ["what-if-a", "what-if-b"]
        return self.state_params[view]

    def set_instance(self, view, index, values):
        assert view in ["instances-a", "instances-b"]
        df = self.state_params[view].to_pd(copy=False)
        df.iloc[index, :] = values

    def get_instance(self, view, index):
        assert view in ["instances-a", "instances-b"]
        return self.state_params[view].iloc(index)

    def has_explanations(self):
        return len(self.state_params["what-if-a"]) > 0

    def set_display_plots(self, plots):
        self.state_params["display_plots"] = plots

    def get_display_plots(self):
        return self.state_params["display_plots"]

    def set_display_instance(self, index):
        self.state_params["display_instance"] = index

    def get_display_instance(self):
        return self.state_params["display_instance"]

    def set_param(self, param, value):
        self.state_params[param] = value

    def get_param(self, param):
        return self.state_params[param]

    def get_feature_values(self):
        return self.features

    def is_tabular(self):
        return isinstance(self.instances, Tabular)

    def _extract_feature_values(self):
        from ..preprocessing.base import Identity
        from ..preprocessing.encode import KBins
        from ..preprocessing.tabular import TabularTransform

        if self.explainer is None:
            return None
        training_data = self.explainer.data
        transformer = TabularTransform(
            cate_transform=Identity(), cont_transform=KBins(n_bins=10)
        ).fit(training_data)
        df = transformer.invert(
            transformer.transform(training_data)
        ).to_pd(copy=False)

        features = OrderedDict()
        for col in training_data.feature_columns:
            features[col] = list(set(df[col].values))
        return features


def init():
    global state
    global whatif_state
    state = State()
    whatif_state = WhatifState()
