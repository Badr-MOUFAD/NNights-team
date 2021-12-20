import pandas as pd


class Experiment():
    def __init__(self, data) -> None:
        self.data = data
        self.default_x_columns = ['x1', 'x2', 'x3']
        pass

    def enrich_jobs(self,  config: dict) -> pd.DataFrame:
        data_copy = self.data.copy()

        return data_copy

    def model(self, config: dict):
        pass

    def run(self, config: dict, use_cache=False) -> pd.DataFrame:
        self.x_columns = get('x_columns', init_x_columns)
        # step 1 : enrich data
        config_enrich = config.get('enrich', None)
        if config_enrich:
            data = self.enrich_jobs(config_enrich)
        # step 2 : model
        config_model = config.get('model', None)
        if config_model:
            model = self.model(config_model)
