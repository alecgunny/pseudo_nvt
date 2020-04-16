import pandas as pd


class Dataset:
    def __init__(self, file_name, batch_size):
        self.df = pd.read_csv(file_name)
        self.batch_size = batch_size
        self.idx = 0
        self.workflow = None
        self.stats_context = None

    def __len__(self):
        return (len(self.df) - 1) // self.batch_size + 1

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self):
            raise StopIteration

        x = self.df.iloc[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
        if self.workflow is not None:
            x = self.workflow.apply(x, stats_context=self.stats_context)

        self.idx += 1
        return x

    def map(self, workflow, stats_context=None):
        self.workflow = workflow
        self.stats_context = stats_context
