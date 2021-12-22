import pandas as pd
import pprint
from nnights.enrich_jobs import dict_enrich
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split


class Experiment():
    """summary."""

    def __init__(self, data) -> None:  # noqa
        self.data = data

        self.meta = {'cache': {'data': None,
                               'x_columns': []},
                     'data': data
                     }

        pass

    def enrich_jobs(self, config: dict, is_inference=False, data=None,):
        if is_inference:
            data_copy = data
        else:
            data_copy = self.data.copy()

        new_columns = []
        for key in config['external_enrich']:

            enrich = dict_enrich[key]
            new_cols = enrich(data_copy)
            new_columns.extend(new_cols)

        for key in config['internal_enrich']:

            enrich = dict_enrich[key]
            new_cols = enrich(data_copy)
            new_columns.extend(new_cols)

        return data_copy, new_columns

    def model(self, data, x_columns, config: dict):
        pprint.pprint(config)
        test_size = config.get('test_size', 0.2)
        # prepare X,y
        X = data[x_columns]
        y = data['target']
        # split data int train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model = XGBRegressor()
        # fit
        model.fit(X_train, y_train, verbose=False)
       # score model
        score = model.score(X_test, y_test)
        print(f'Score : {score}')

        pass

    def run(self, config: dict, use_cache=False) -> pd.DataFrame:
        if use_cache:  # skip enrichment
            pass

        else:
            # step 1 : enrich data
            config_enrich = config.get('enrich', None)
            x_cols = config.get('x_columns', [])
            if config_enrich:
                enriched_data, new_cols = self.enrich_jobs(config_enrich)
                # cache element
                self.meta['cache']['data'] = enriched_data
                self.meta['cache']['x_columns'] = x_cols + new_cols
            else:
                self.meta['cache']['data'] = self.data
                self.meta['cache']['x_columns'] = x_cols

        # extract data and x_columns
        data = self.meta['cache']['data']
        x_columns = self.meta['cache']['x_columns']

        # step 2 : model
        config_model = config.get('model', None)
        if config_model:
            model = self.model(data, x_columns, config_model)
