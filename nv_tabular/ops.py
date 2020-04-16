import six
import re
import numpy as np
import pandas as pd

from . import namedtuple, utils, stats


class Op(namedtuple(
        'Op',
        ['columns', 'replace', 'name'],
        defaults=[None, True, None])):
    '''
    Stateless representation of a feature operation in a workflow. If `columns`
    is None, the Op is generic in the sense that it will be applied to all
    the columns in the class given by `default_in` in the Workflow to any
    Workflow to which it is added. Providing an argument for `columns` means
    that the op will only be applied to those columns, and adding such an op
    to a workflow will check to make sure that the appropriate columns are
    present.
    Parameters
    ----------------------
    columns: None, list(str), or callable
        Column names that this op will act on. If left as None op will act
        on whatever columns provided by workflow that fall under the class
        prescribed by the `default_in` attribute. Columns can be enumerated
        explicitly in a list, or use a callable function which will map from
        column name strings to a boolean indicating whether such a column
        should be acted upon by the op
    replace: bool
        Whether the Op will act in-place or not. In-place ops will ignore
        `name` and return the transformed columns as they were originally
        named.
    name: None, str, or callable
        Name to give to this op. Ignored if `replace=True`. If left as None,
        column names acted on by the op will have the snake case version of the
        op class name appended with an underscore. If a string is provided,
        that will be the name appended with an underscore. Otherwise a callable
        will map from the original column name to the new column name
    '''
    # override this attribute to specify the category of columns that
    # an op should apply to if specific columns aren't specified
    __default_in = utils.VariableTypes.ALL
    @property
    def default_in(self):
        return self.__default_in

    # override this property if the op requires dataset statistics in order
    # to execute _op_logic. Leaving it this way instead of using privtate
    # attribute in case subclass wants to use __init__ args to decide
    # which stats are necessary
    @property
    def stats_required(self):
        return []

    @property
    def _id(self):
        '''
        if an explicit name was specified, use this as the _id
        Otherwise return the snake case version of the class name
        '''
        if six.callable(self.name) or self.name is None:
            return utils.snake_case_class_name(self)
        return self.name

    def apply(self, gdf, stats_context=None, whitelist=None):
        '''
        workhorse method responsible for applying the op's function
        '''
        columns = self._validate_columns(gdf.columns, whitelist=whitelist)
        stat_state = self._validate_stats(stats_context)

        new_gdf = self._op_logic(gdf[columns], stat_state)
  
        if self.replace:
            gdf[columns] = new_gdf
        else:
            gdf[self.get_output_names(columns)] = new_gdf
        return gdf

    def _validate_columns(self, columns, whitelist=None):
        if six.callable(self.columns):
            if not any([self.columns(column) for column in columns]):
                raise ValueError(
                    'Op {} not set to act on any columns in {}'.format(
                        self._id, ', '.join(columns))
                )
            return [column for column in columns if self.columns(column)]

        elif self.columns is not None:
            missing_columns = []
            for column in self.columns:
                if (column not in columns and
                    (whitelist is None or column not in whitelist)):
                    missing_columns.append(column)
            if missing_columns:
                raise ValueError(
                    'Op {} was called on columns {}, which does not include'
                    'columns {} op has been defined for'.format(
                        self._id, 
                        ', '.join(columns),
                        ', '.join(missing_columns))
                )
            return [
                column for column in self.columns if 
                    whitelist is None or column not in whitelist
            ]
        else:
            return columns

    def _validate_stats(self, stats_context):
        '''
        internal method for checking that the provided stats_context has been
        properly initialized with the necessary values to execute the op
        '''
        if self.stats_required:
            if stats_context is None:
                raise ValueError(
                    'No stats context was provided when applying op {}'
                    'which requires stats {}'.format(
                        self._id,
                        ', '.join([stat._id for stat in self.stats_required]))
                )
            op_stats = stats_context.state.get(self._id)
            if op_stats is None:
                raise ValueError(
                    'Stats context provided to op {} has no corresponding'
                    'calculated stats'.format(self._id)
                )
#             missing_stats = []
#            for stat in self.stats_required:
#                if not any(
#                        [isinstance(_stat, type(stat)) for _stat in op_stats]):
#                 
#                    missing_stats.append(stat)
            # TODO: be able to check for specific missing stuff
            missing_stats = len(self.stats_required) != len(op_stats)
            if missing_stats:
                missing_stats = self.stats_required
                raise ValueError(
                    'Stats context provided to op {} missing necessary'
                    'stat values {}'.format(
                        ', '.join([stat._id for stat in missing_stats]))
                )
            return op_stats

    def map_column_name(self, column_name):
        '''
        maps an input column name to the name of the column returned by
        the op. If the given column name wouldn't be acted upon by the op,
        returns None.
        '''
        if self.columns is not None:
            if six.callable(self.columns):
              if not self.columns(column_name):
                  return None
            else:
              if column_name not in self.columns:
                  return None

        if self.replace:
            return column_name
        if six.callable(self.name):
            return self.name(column_name)
        return '_'.join([column_name, self._id])

    def get_output_columns(self, columns):
        mapped = [self.map_column_name(column) or column for column in columns]
        if self.replace:
            return mapped
        else:
            # TODO: do we need deterministic behavior here?
            return list(set(mapped) | set(columns))

    def __call__(self, workflow):
        assert isinstance(workflow, Workflow)
        self._validate_columns(workflow.columns)

        if hasattr(self, 'ops'):
            # this is a workflow, so just add all the workflow's ops
            new_ops = self.ops
        elif self.columns is None:
            # this has no columns, so set columns of the appropriate
            # variable type from the workflow as the columns for a new op
            columns = workflow.get_columns(self.default_in)
            new_ops = [self._replace(columns=columns)]
        else:
            # this is an op and it has columns, so use it as is
            new_ops = [self]

        # TODO: similar checking of _id that will add a replaced version of
        # self if another op has the same _id. Maybe we do away with the
        # name callability
        return workflow._replace(ops=workflow.ops+new_ops)


class Workflow(
        namedtuple(
            'Workflow',
            ['cat_names', 'cont_names', 'label_names', 'ops'],
            defaults=[[], [], [], []]),
        Op):
    def __new__(cls, cat_names=[], cont_names=[], label_names=[], ops=[]):
        obj = cls.__bases__[0].__new__(
            cls, cat_names, cont_names, label_names, [])
        for op in ops:
          obj = op(obj)
        return obj

    def get_columns(self, variable_type):
        assert hasattr(utils.VariableTypes, variable_type)
        internal_name = re.match('C(A|ON)T', variable_type).group().lower()
        columns = getattr(self, '{}_names'.format(internal_name))

        for op in self.ops:
            if op.default_in.lower().startswith(variable_type):
                columns = op.get_output_columns(columns)
        return columns

    @property
    def categorical_columns(self):
        return self.get_columns(utils.VariableTypes.CATEGORICAL)

    @property
    def continuous_columns(self):
        return self.get_columns(utils.VariableTypes.CONTINUOUS)

    @property
    def columns(self):
        return (
            self.categorical_columns +
            self.continuous_columns +
            self.label_names
        )

    def apply(self, gdf, stats_context=None):
        for op in self.ops:
            gdf = op.apply(gdf, stats_context)
        return gdf


class Log(Op):
  default_in = utils.VariableTypes.CONTINUOUS
  def _op_logic(self, gdf, stats_context=None):
      return np.log(gdf)


class Normalize(Op):
    default_in = utils.VariableTypes.CONTINUOUS
    @property
    def stats_required(self):
        return [stats.Moments()]

    def _op_logic(self, gdf, stats):
        means = np.array([stats[0][column]['mean'] for column in gdf.columns])
        vars_ = np.array([stats[0][column]['var'] for column in gdf.columns])
        return (gdf - means) / np.sqrt(vars_)


class Categorify(Op):
    default_in = utils.VariableTypes.CATEGORICAL
    @property
    def stats_required(self):
        return [stats.DLLabelEncoder()]

    def _op_logic(self, gdf, stats):
        new_gdf = pd.DataFrame(index=gdf.index)
        for column in gdf.columns:
            new_gdf[column] = stats[0][column]['encoder'].loc[gdf[column]].values
        return new_gdf
