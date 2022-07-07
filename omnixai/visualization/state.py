class State:

    def __init__(self):
        self.class_names = None
        self.params = None
        self.instances = None
        self.instance_indices = []

        views = ["local", "global", "prediction", "data"]
        self.explanations = {view: {} for view in views}
        self.num_figures_per_row = {view: 2 for view in views}
        self.plots = {view: [] for view in views}
        self.display_plots = {view: [] for view in views}
        self.display_instance = {"local": 0}

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
        self.instance_indices = list(range(len(self.instances))) if instances is not None else []

        self.set_explanations("local", local_explanations)
        self.set_explanations("global", global_explanations)
        self.set_explanations("data", data_explanations)
        self.set_explanations("prediction", prediction_explanations)
        for view, explanations in self.explanations.items():
            self.plots[view] = [name for name in explanations.keys()]
            self.display_plots[view] = self.plots[view]

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
        assert view in self.num_figures_per_row
        self.num_figures_per_row[view] = n

    def get_num_figures_per_row(self, view):
        return self.num_figures_per_row[view]

    def set_plots(self, view, plots):
        assert view in self.plots
        self.plots[view] = plots

    def get_plots(self, view):
        return self.plots[view]

    def set_display_plots(self, view, plots):
        assert view in self.display_plots
        self.display_plots[view] = plots

    def get_display_plots(self, view):
        return self.display_plots[view]

    def set_display_instance(self, view, index):
        assert view in self.display_instance
        self.display_instance[view] = index

    def get_display_instance(self, view):
        return self.display_instance[view]


def init():
    global state
    state = State()
