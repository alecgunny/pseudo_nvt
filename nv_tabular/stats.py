from . import namedtuple
from .utils import snake_case_class_name

from collections import defaultdict
import numpy as np
import pandas as pd


class Stat:
    @property
    def state_values(self):
        return []

    @property
    def _id(self):
        return snake_case_class_name(self)

    def initialize_state(self):
        raise NotImplementedError

    def update_value(self, state, column):
        raise NotImplementedError


class Moments(Stat):
    @property
    def state_values(self):
        return ['count', 'mean', 'var']

    def initialize_state(self):
        return {value: 0 for value in self.state_values}

    def update_state(self, state, column):
        old_count = state['count']
        this_count = column.shape[0]
        new_count = old_count + this_count

        old_mean = state['mean']
        this_mean = np.mean(column)
        new_mean = (
            this_mean*(this_count/new_count) +
            old_mean*(old_count/new_count)
        )

        old_var = state['var']
        this_var = np.var(column)
        new_var = (
            this_var*(this_count/new_count) +
            old_var*(old_count/new_count) +
            (old_count*this_count/new_count**2) * (this_mean - old_mean)**2
        )
        return {'count': new_count, 'mean': new_mean, 'var': new_var}


class DLLabelEncoder(Stat):
    @property
    def state_values(self):
        return ['encoder']

    def initialize_state(self):
        return {'encoder': pd.Series()}

    def update_state(self, state, column):
        new_index = np.unique(np.concatenate([
            state['encoder'].index, column.unique()]))
        return {
            'encoder': pd.Series(np.arange(len(new_index)), index=new_index)
        }


class StatsContext(namedtuple('StatsContext', 'workflow')):
    __state = {}
    @property
    def state(self):
        return self.__state

    def fit(self, dataset, warm_start=False):
        if warm_start and len(self.state) > 0:
            state = self.state.copy()
            for op in self.workflow.ops:
                assert op._id in state
                for column in op.columns:
                    assert column in state[op._id]
        else:
            state = defaultdict(list)
            for op in self.workflow.ops:
                for stat in op.stats_required:
                    state[op._id].append(
                        {column: stat.initialize_state() for
                            column in op.columns}
                    )

        for gdf in dataset:
            whitelist = []
            for op in self.workflow.ops:
                if op._id in state:
                    for stat, stat_state in zip(
                          op.stats_required, state[op._id]):
                        for column, column_state in stat_state.items():
                          if column not in gdf.columns:
                              # TODO: make this more explicit
                              raise ValueError(
                                  'Stat required for column that relied on '
                                  'stat earlier in workflow'
                              )
                          stat_state[column] = stat.update_state(
                              column_state, gdf[column])
                    whitelist.extend(op.columns)
                    gdf = gdf.drop(columns=op.columns)
                else:
                    gdf = op.apply(gdf, None, whitelist)
        self.__state = state
