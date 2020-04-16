import nv_tabular as nvt
import pickle

CATEGORICAL_COLUMNS = ['uid', 'iid', 'location', 'has_interacted_before']
CONTINUOUS_COLUMNS = ['timestamp', 'user_age', 'item_average_rating']
LABEL_COLUMNS = ['click', 'purchase']

# PREPROCESSING STAGE

# workflow is a stateless composition of phases
workflow = nvt.workflow.Workflow(
    cat_names=CATEGORICAL_COLUMNS,
    cont_names=CONTINUOUS_COLUMNS,
    label_names=LABEL_COLUMNS
)

# phase is stateless composition of ops
continuous_normalize_phase = nvt.workflow.Phase(
    variable_type='continuous',
    name='log_and_normalize',
    ops=[
        nvt.ops.Log(),
        nvt.ops.Normalize()
    ]
)
# callable on a workflow to return a new workflow that includes this new phase
workflow = continuous_normalize_phase(workflow)

# do the same for a categorical phase
categorical_encoding_phase = nvt.workflow.Phase(
    variable_type='categorical',
    name='categorical_encode',
    ops=[nvt.ops.Categorify(columns=['uid', 'iid', 'location'])]
)
workflow = categorical_encoding_phase(workflow)

# dataset is a symbolic representation of some specific set of samples
# from the desired dataset schema
dataset = nvt.dataset()

# stats context is a stateful representation of the statistics needed
# by the ops in a workflow
stats_context = nvt.stats.StatsContext(workflow)

# state is established by fitting it to a particular set of values
stats_context.fit(dataset)

# map method applies a workflow at each iteration of the dataset
# uses stats recorded by an initialized stats_context
dataset = dataset.map(workflow, stats_context=stats_context)

# writer is a simple object for iterating through a dataset, optionally
# shuffling somehow, then writing the results
writer = nvt.writer.Writer('/path/to/write/dataset/to.parquet')
writer.write(dataset)

# workflow and stats_context are pickleable so that they can be retrieved
# later
with open('/path/to/write/workflow/to.pickle', 'wb') as f:
    pickle.dump(workflow, f)
with open('/path/to/write/stats/context/to.pickle', 'wb') as f:
    pickle.dump(stats_context, f)