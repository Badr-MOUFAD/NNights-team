# @title Default title text
"""[summary]."""

from typing import List
import numpy as np
import pandas as pd
import json
import joblib

from nnights.enrich_jobs import dict_enrich

from sklearn.ensemble import GradientBoostingRegressor as Gb_regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class Experiment:
    """[summary]."""

    def __init__(self, name, data) -> None:  # noqa
        self.data = data
        self.name = name

        self.meta = {
            'cache': {
                'data': None,
                'x_columns': [],
                'model': None,
                'scaler': None
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
        for key in config:
            print(key, ' ...')
            enrich = dict_enrich[key]
            new_cols = enrich(data_copy)
            new_columns.extend(new_cols)

        return data_copy, new_columns

    def cross_validate_model(self, model, X, y):
        # cross val with rmse
        nb_folds = 10
        scores = -cross_val_score(model, X, y, cv=nb_folds,
                                  scoring="neg_root_mean_squared_error")

        props_std_plot = dict(
            mode="lines",
            line_width=1,
            marker_color='#EF553B',
        )

        mean_scores = scores.mean()
        std_scores = scores.std()

        fig = go.Figure(
            data=[
                go.Scatter(y=scores, name="rmse"),
                go.Scatter(
                    y=[mean_scores for _ in range(nb_folds)],
                    mode="lines",
                    marker_color='#EF553B',
                    name="mean"
                ),
                go.Scatter(
                    y=[mean_scores + std_scores for _ in range(nb_folds)],
                    showlegend=False,
                    **props_std_plot
                ),
                go.Scatter(
                    y=[mean_scores - std_scores for _ in range(nb_folds)],
                    fill='tonexty',
                    name="std",
                    **props_std_plot
                ),
            ]
        )

        fig.update_layout(
            title='Result cross validation',
            yaxis_title="score"
        )

        fig.show()
        return

    def get_feat_imporance(self, model, x_columns):
        feat_imp = dict(zip(x_columns, model.feature_importances_))
        sorted_feat_imp = {k: v for k, v in sorted(
            feat_imp.items(), key=lambda item: item[1])}
        return sorted_feat_imp

    def model_data(
        self,
        data: pd.DataFrame,
        x_columns: List[str],
        config: dict
    ):
        """[summary]."""
        # destruct config.
        default_params = {'objective': 'reg:squarederror'}
        model_params = config['model_params'].get(
            'model_params', default_params)
        use_cv = config['train_params'].get('use_cv', False)

        # prepare data
        X = data[x_columns]
        y = data['target']

        # scale data
        scale: dict = config.get("scale", None)
        if scale:
            # params
            li_features = scale["li_features"]

            # init feature scaler
            col_scaler = ColumnTransformer(
                transformers=[
                    ('scaled', StandardScaler(), li_features)
                ]
            )

            # fit and save
            col_scaler.fit(X)
            scaler_cache = {"li_features": li_features,
                            "col_scaler": col_scaler}
            self.meta["cache"]["scaler"] = scaler_cache

            # transform data
            X[li_features] = pd.DataFrame(col_scaler.transform(X))

        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123654)

        # init model
        print('> fit model ...')
        xgbr = RandomForestRegressor()  # xgb.XGBRegressor(**model_params)
        print('model :', xgbr)

        # cross-val
        if use_cv:
            print('> cv results : ')
            self.cross_validate_model(xgbr, X_train, y_train)

        # fit model
        xgbr.fit(X_train, y_train)

        # score on validation
        predictions_test = xgbr.predict(X_test)
        predictions_train = xgbr.predict(X_train)

        error_train = mean_squared_error(
            predictions_train, y_train, squared=False)
        error_test = mean_squared_error(
            predictions_test, y_test, squared=False)

        print('> score model ...\n'
              f"RMSE on train : {error_train}\n"
              f"RMSE on test : {error_test}"
              )

        # feat importance.
        # print('--Feat imporance  ...')
        # print(' ')
        # feat_imporatance = self.get_feat_imporance(xgbr, x_columns)
        # # pprint.pprint(feat_imporatance)
        # plt.bar(list(feat_imporatance.keys()), list(feat_imporatance.values()))
        # plt.xticks(rotation=90)
        # plt.show()

        # log
        self.meta['cache']['model'] = xgbr

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
                print('-- Enrich start ------------- ')
                enriched_data, new_cols = self.enrich_jobs(config_enrich)

                # cache element
                self.meta['cache']['data'] = enriched_data
                self.meta['cache']['x_columns'] = x_cols + new_cols
                # store enrichment config for inference reproduction
                self.meta['cache']['enrich_config'] = config_enrich
            else:
                self.meta['cache']['data'] = self.data
                self.meta['cache']['x_columns'] = x_cols

        # extract data and x_columns
        data = self.meta['cache']['data']
        x_columns = self.meta['cache']['x_columns']

        # step 2 : model
        config_model = config.get('model', None)
        if config_model:
            print('-- Model start -------------')
            print('x_columns : ', x_columns)
            self.model_data(data, x_columns, config_model)

    def generate_submission(self, X_data):
        # get enrich config.
        enrich_config = self.meta['cache']['enrich_config']

        # enrich data
        X_data_enrich, _ = self.enrich_jobs(
            enrich_config, data=X_data, is_inference=True)

        # get model and x_cols from cache
        model = self.meta['cache']['model']
        x_columns = self.meta['cache']['x_columns']

        # scale
        cache_scaler = self.meta["cache"]["scaler"]
        if cache_scaler:
            li_features = cache_scaler["li_features"]
            col_scaler = cache_scaler["col_scaler"]

            # scale data
            X_data_enrich[li_features] = pd.DataFrame(
                col_scaler(X_data_enrich))

        # predict on data
        predictions = model.predict(X_data_enrich[x_columns])
        print(predictions[:5])
        # create df
        submission = pd.DataFrame(predictions)
        return submission

    def freeze(self, path, X_data, with_sub=False):
        """[summary]."""
        # create folder by exp name
        import os
        path = path+'/'+self.name

        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

        # pickle model
        # get model
        model = self.meta['cache']['model']
        model_path = path+'/'+'model.joblib.dat'
        joblib.dump(model, model_path)
        # store metadata

        meta = {
            'date': '',
            'config_enrich': self.meta['cache']['enrich_config'],
            'x_columns': self.meta['cache']['x_columns']
        }

        meta_path = path+'/'+'metadata.json'
        with open(meta_path, "w") as outfile:
            json.dump(meta, outfile)

        # generate submission
        if with_sub:
            submission = self.generate_submission(X_data)
            sub_path = path + '/'+'submission.csv'
            print('generate submission ', sub_path)
            # to csv
            submission.to_csv(sub_path, index=False, header=False)


# feature importance util
# exp diff on previous model got from freeze
