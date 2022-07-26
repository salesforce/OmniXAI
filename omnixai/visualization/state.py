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
            self.set_num_figures_per_row(view, 2)
            self.set_display_instance(view, 0)

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


def init():
    global state
    state = State()
