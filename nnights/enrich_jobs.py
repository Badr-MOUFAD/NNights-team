# @title
# enrich_jobs.py

"""Jobs run on data to enrich it.
A job is a function that takes the flight data frame as inputs
add columns to it and then outputs theses added columns.
"""

from typing import List

import pandas as pd
from .utils import encode_location

from .utils import (is_holiday, distance_next_holiday,
                    distance_previous_holiday,
                    distance_to_holidays)


def add_is_holiday(df: pd.DataFrame) -> List[str]:
    """Determine if dates are holidays.
    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.
    Returns
    -------
    List[str]
        returns list of added columns.
    """
    df["is_holiday"] = df["flight_date"].apply(is_holiday)

    new_cols = ["is_holiday"]
    return new_cols


def add_distance_to_next_holiday(df: pd.DataFrame) -> List[str]:
    """Compute distance to the next holiday.
    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.
    Returns
    -------
    List[str]
        returns list of added columns.
    """
    df["distance_to_next_holiday"] = df["flight_date"].apply(
        distance_next_holiday)

    new_cols = ["distance_to_next_holiday"]
    return new_cols


def add_distance_to_previous_holiday(df: pd.DataFrame) -> List[str]:
    """Compute the distance to the previous holiday.
    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.
    Returns
    -------
    List[str]
        returns list of added columns.
    """
    df["distance_to_previous_holiday"] = df["flight_date"].apply(
        distance_previous_holiday)

    new_cols = ["distance_to_previous_holiday"]
    return new_cols


def add_distance_to_holidays(df: pd.DataFrame) -> List[str]:
    """Compute the distance between dates and holidays.
    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.
    Returns
    -------
    List[str]
        returns list of added columns.
    """
    # compute distance to holidays
    df_distances = pd.concat(
        [distance_to_holidays(d) for d in df["flight_date"]],
        axis=1,
    ).T

    # add to df
    new_cols = []
    for col in df_distances.columns:
        new_col = f"distance_to_{col}"
        df[new_col] = df_distances[col]
        new_cols.append(new_col)

    return new_cols


def add_day_of_year(df: pd.DataFrame) -> List[str]:
    """Add day of year to data frame.

    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.

    Returns
    -------
    List[str]
        returns list of added columns.
    """
    # flight_date to date
    df['flight_date'] = pd.to_datetime(df['flight_date'])
    df["day_of_year"] = df.apply(
        lambda x: x["flight_date"].dayofyear,
        axis=1
    )
    # get new cols
    new_cols = ['day_of_year']

    return new_cols


def encode_loc(df: pd.DataFrame) -> List[str]:
    """Encode.

    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.

    Returns
    -------
    List[str]
        returns list of added columns.
    """
    df['from_enc'] = encode_location(df['from'])
    df['to_enc'] = encode_location(df['to'])

    new_cols = ['from_enc', 'to_enc']
    return new_cols


dict_enrich = {
    'add_is_holiday': add_is_holiday,
    'add_distance_to_next_holiday': add_distance_to_next_holiday,
    'add_distance_to_previous_holiday': add_distance_to_previous_holiday,
    'add_distance_to_holidays': add_distance_to_holidays,
    'add_day_of_year': add_day_of_year,
    'encode_locations': encode_loc,
}
