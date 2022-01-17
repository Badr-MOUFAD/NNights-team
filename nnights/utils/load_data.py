
import pandas as pd

# params
DATA_TRAIN_PATH = "data/flights_train.csv"
DATA_TEST_PATH = "data/flights_Xtest.csv"


def load_flights(dtype: str = 'train') -> pd.DataFrame:
    """[summary].

    Parameters
    ----------
    dtype : str, optional
        [description], by default 'train'

    Returns
    -------
    pd.DataFrame
        [description]
    """
    path = DATA_TRAIN_PATH if dtype == 'train' else DATA_TEST_PATH

    # load flight data
    flights = pd.read_csv(
        path,
        parse_dates=["flight_date"]
    )
    # sort
    flights.sort_values(
        by=["flight_date"],
        inplace=True
    )

    return flights
