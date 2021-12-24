# test
import pandas as pd
from nnights.experiment import Experiment
data_train = pd.read_csv('../data/flights_train.csv')
# init exp
exp = Experiment(name='enrich_with_holiday', data=data_train)

config = {'enrich': ['add_day_of_year', 'encode_locations', 'add_distance_to_holidays'],
          'model': {'model_params': {'objective': 'reg:squarederror'},
                    'train_params': {'use_cv': True}},
          'x_columns': ['avg_weeks', 'std_weeks']}

# run exp
exp.run(config)
# or if enriched run instead exp.run(config,use_cache=True)


# store exp data
X_data = pd.read_csv('../data/flights_Xtest.csv')
exp.freeze(path='', X_data=X_data, with_sub=True)
