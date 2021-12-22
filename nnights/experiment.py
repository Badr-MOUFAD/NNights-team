"""[summary]."""

from typing import List
import numpy as np
import pandas as pd

from nnights.enrich_jobs import dict_enrich

from sklearn.ensemble import GradientBoostingRegressor as Gb_regressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Experiment():
    """summary."""

    def __init__(self, name, data) -> None:  # noqa
        self.data = data
        self.name = name

        self.meta = {
            'cache': {
                'data': None,
                'x_columns': []
            },
            'exp_name': self.name
        }

        pass

    def enrich_jobs(
        self,
        config: dict,
        is_inference: bool = False,
        data: pd.DataFrame = None
    ):
        """[summary]."""
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

    def model(
        self,
        data: pd.DataFrame,
        x_columns: List[str],
        config: dict
    ):
        """[summary]."""
        test_size = config.get('test_size', 0.2)
        # prepare X,y
        X = data[x_columns]
        y = data['target']
        # split data int train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)

        model = Gb_regressor()
        # fit
        model.fit(X_train, y_train)
        # compute rmse on test data
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print('rmse on test', rmse)
        self.model = model
        pass

    def run(
        self,
        config: dict,
        use_cache: bool = False
    ) -> pd.DataFrame:
        """[summary]."""
        if use_cache:  # skip enrichment
            pass

        else:
            # step 1 : enrich data
            config_enrich = config.get('enrich', None)
            x_cols = config.get('x_columns', [])
            if config_enrich:
                print('enrich start ... ')
                enriched_data, new_cols = self.enrich_jobs(config_enrich)

                # cache element
                self.meta['cache']['data'] = enriched_data
                self.meta['cache']['x_columns'] = x_cols + new_cols
            else:
                self.meta['cache']['data'] = self.data
                self.meta['cache']['x_columns'] = x_cols
                # store enrichment config for inference reproduction
                self.meta['cache']['enrich_config'] = config_enrich

        # extract data and x_columns
        # data = self.meta['cache']['data']
        x_columns = self.meta['cache']['x_columns']

        # step 2 : model
        config_model = config.get('model', None)
        if config_model:
            print('model start ...')
            print('x_columns : ', x_columns)
            # print(x_columns)
            # model = self.model(data, x_columns, config_model)

    def freeze(self):
        """[summary]."""
        # enrich_config = self.meta['cache']['enrich_config']
        # model_inspection = {}
        # model = self.model
        # todo
        # pickle model
        # get model meta (feat imporatnce amd score)


# feature importance util
# exp diff on previous model got from freeze
