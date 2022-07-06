
class State:

    def __init__(self):
        # Local explanation
        self.instances = None
        self.local_explanations = {}
        self.instance_indices = []
        self.show_instance = 0
        # Global explanation
        self.global_explanations = {}

        self.class_names = None
        self.params = None
        self.num_figures_per_row = {
            "local": 2,
            "global": 2
        }
        self.plots = []
        self.show_plots = []

    def set(self, instances, local_explanations, global_explanations, class_names, params):
        self.instances = instances
        self.instance_indices = list(range(len(self.instances))) if instances is not None else []
        self.local_explanations = local_explanations if local_explanations is not None else {}
        self.global_explanations = global_explanations if global_explanations is not None else {}
        self.class_names = class_names
        self.params = {} if params is None else params
        self.plots = [name for name in self.local_explanations.keys()] + [
            f"{name}:global" for name in self.global_explanations.keys()
        ]
        self.show_plots = self.plots

    def has_explanations(self):
        if len(self.local_explanations) > 0 or len(self.global_explanations) > 0:
            return True
        else:
            return False

    def set_num_figures_per_row(self, view, n):
        assert view in self.num_figures_per_row
        self.num_figures_per_row[view] = n

    def get_num_figures_per_row(self, view):
        return self.num_figures_per_row[view]


def init():
    global state
    state = State()
