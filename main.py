import nv_tabular as nvt
import pickle

CATEGORICAL_COLUMNS = ['uid', 'iid', 'location', 'has_interacted_before']
CONTINUOUS_COLUMNS = ['timestamp', 'user_age', 'item_average_rating']
LABEL_COLUMNS = ['click', 'purchase']

# PREPROCESSING STAGE

# workflow is an op that contains collections of ops, which can be
# instantiated into the workflow
workflow = nvt.ops.Workflow(
    cat_names=CATEGORICAL_COLUMNS,
    cont_names=CONTINUOUS_COLUMNS,
    label_names=LABEL_COLUMNS,
    ops=[]
)

# or added via calling
workflow = nvt.ops.Categorify(columns=['uid', 'iid', 'location'])(workflow)
print(workflow.ops)

# can even call workflows on other workflows
continuous_workflow = nvt.ops.Workflow(
    cont_names=CONTINUOUS_COLUMNS,
    ops=[nvt.ops.Log(), nvt.ops.Normalize()]
)
workflow = continuous_workflow(workflow)
print(workflow.ops)

# dataset is a symbolic representation of some specific set of samples
# from the desired dataset schema
dataset = nvt.dataset('this_data.csv', batch_size=128)

# stats context is a stateful representation of the statistics needed
# by the ops in a workflow
stats_context = nvt.stats.StatsContext(workflow)

# state is established by fitting it to a particular set of values
stats_context.fit(dataset)
print(stats_context.state['categorify'][0]['location']['encoder'])

# map method applies a workflow at each iteration of the dataset
# uses stats recorded by an initialized stats_context
dataset.map(workflow, stats_context=stats_context)

# writer is a simple object for iterating through a dataset, optionally
# shuffling somehow, then writing the results
writer = nvt.writer.Writer('this_data.parquet')

# won't write because I don't have a parquet engine
# writer.write(dataset)

# workflow and stats_context are pickleable so that they can be retrieved
# later
with open('workflow.pickle', 'wb') as f:
    pickle.dump(workflow, f)
with open('stats.pickle', 'wb') as f:
    pickle.dump(stats_context, f)

del dataset, stats_context, workflow

with open('workflow.pickle', 'rb') as f:
    workflow = pickle.load(f)
with open('stats.pickle', 'rb') as f:
    stats_context = pickle.load(f)

dataset = nvt.dataset('this_data.csv', batch_size=128)
dataset.map(workflow, stats_context=stats_context)
x = next(iter(dataset))
print(x)
