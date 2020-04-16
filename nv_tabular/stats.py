from . import namedtuple
from .utils import snake_case_class_name
import numpy as np


class Stat:
    @property
    def state_values(self):
        return []

    @property
    def _id(self):
        return snake_case_class_name(self)

    def update_value(self, state, column):
        raise NotImplementedError


class Mean(Stat):
    @property
    def state_values(self):
        return ['count', 'mean']

    def initialize_state(self):
        return {value: 0 for value in self.state_values}

    def update_state(self, state, column):
        old_count = state['count']
        new_count = old_count + column.shape[0]

        old_mean = state['mean']
        this_mean = np.mean(column)
        new_mean = (
            this_mean*(column.shape[0]/new_count) +
            old_mean*(old_count/new_count)
        )
        return {'count': new_count, 'mean': new_mean}


class Std(Stat):
    @property
    def state_values(self):
        return ['count', 'std']

    def update_value(self, state, column):
        # TODO: implement
        pass


class StatState:
    def __init__(self, stat):
        self.stat = stat
        self.__state = stat.initialize_state()

    @property
    def state(self):
        return self.__state

    def fit_partial(self, column):
        self.__state = self.stat.update_state(self.state, column)


class StatsContext(namedtuple('StatsContext', 'workflow')):
    __state = {}
    @property
    def state(self):
        return self.__state

    def fit(self, dataset, warm_start=False):
        if warm_start and len(self.state) > 0:
            state = self.state.copy()

        else:
            state = {}
            for phase in self.workflow.phases:
                for op in phase.ops:
                    if len(op.stats_required) > 0:
                        state[op._id] = [
                            StatState(stat) for stat in op.stats_required]

#        for gdf in dataset:
#            for phase in workflow.phases:
#                if not op._id in state:
#            for op, stats in state.items():
#                pass
        self.__state = state
