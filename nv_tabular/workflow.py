from . import namedtuple


class Phase(namedtuple(
        'Phase', ['variable_type', 'ops', 'name'], defaults=[[], None])):
    '''
    Generic container for compositions of ops, which represent its state.
    Only intended to operate on one type of variables (categorical or
    continuous) at a time. Generic in the sense that compositions of ops
    without specified columns can be applied to the set of columns of the
    appropriate variable type in any arbitrary schema. This makes phases
    recyclable between workflows, and constrains them only to the extent
    that their ops are constrained
    '''
    def validate(self, columns):
        columns = [i for i in columns] # making a copy
        for op in self.ops:
            if len(columns) > 0:
                op.validate_columns(columns)
                columns = op.get_output_columns(columns)
            elif op.columns is not None:
                columns = op.get_output_columns(op.columns)

    def get_output_columns(self, columns):
        for op in self.ops:
            columns = op.get_output_columns(columns)
        return columns

    def __call__(self, workflow):
        assert isinstance(workflow, Workflow)
        self.validate(workflow.columns)
        phases = workflow.phases + [self]
        return workflow._replace(phases=phases)


class Workflow(namedtuple(
        'Workflow',
        ['cat_names', 'cont_names', 'label_names', 'phases'],
        defaults=[[], [], [], []])):
    def get_columns_at_phase(self, variable_type, phase=None):
        assert variable_type in ('cat', 'cont')
        columns = getattr(self, '{}_names'.format(variable_type))
        for p in self.phases:
            if p.variable_type == variable_type:
                columns = p.get_output_columns(columns)
            if phase is not None and p.name == phase:
                break
        return columns

    @property
    def categorical_columns(self):
        return self.get_columns_at_phase('cat', phase=None)

    @property
    def continuous_columns(self):
        return self.get_columns_at_phase('cont', phase=None)

    @property
    def columns(self):
        return (
            self.categorical_columns +
            self.continuous_columns +
            self.label_names
        )
