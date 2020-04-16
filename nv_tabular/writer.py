from . import namedtuple

class Writer(namedtuple('Writer', 'write_path')):
    def write(self, dataset, workflow=None, stats_context=None, shuffler=None):
        for gdf in dataset:
            if workflow is not None:
                workflow.apply(gdf, stats_context=stats_context)
            if shuffler is not None:
                shuffler.shuffle(gdf)
            gdf.to_parquet(self.write_path)
