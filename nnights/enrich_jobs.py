"""Jobs run on data to enrich it.

A job is a function that takes the flight data frame as inputs
add columns to it and then outputs theses added columns.
"""

from typing import List
import json

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

    # get new cols
    new_cols = ['from_enc', 'to_enc']

    return new_cols


def special_one_hot_encode(df: pd.DataFrame) -> List[str]:
    """Encode source target like adjacency matrix.

    Parameters
    ----------
    df : pd.DataFrame
        flight dataset.

    Returns
    -------
    List[str]
        returns list of added columns.
    """
    # construct dummies
    dummies_source = pd.get_dummies(df["from"])
    dummies_destination = pd.get_dummies(df["to"])
    df_one_hot_enc = dummies_destination + (-1) * dummies_source

    # append features to df
    for col in df_one_hot_enc.columns:
        df[col] = df_one_hot_enc[col]

    # get new cols
    new_cols = list(df_one_hot_enc.columns)

    return new_cols


def add_path_distance(df: pd.DataFrame) -> List[str]:
    """Add distance (km) col given flight path.

    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.

    Returns
    -------
    List[str]
        returns list of added columns.
    """
    path_distances = pd.read_csv('data/path_distances.csv')
    df['path_distance'] = df.apply(lambda row: path_distances[(path_distances['from'] == row['from']) & (
        path_distances['to'] == row['to'])]['distance'].values[0], axis=1)
    # get new cols
    new_cols = ['path_distance']

    return new_cols


def add_path_embedding(df: pd.DataFrame) -> List[str]:
    """Add degree centrality.

    Parameters
    ----------
    df : pd.DataFrame
        flight data frame.

    Returns
    -------
    List[str]
        returns list of added columns.
    """
    # load degree centrality
    with open('data/in_degree.json') as f:
        in_degree = json.load(f)
    with open('data/out_degree.json') as f:
        out_degree = json.load(f)

    # add in degree  from node
    df['from_in_degree'] = df.apply(lambda row: in_degree[row['from']], axis=1)
    # add out degree  from node
    df['from_out_degree'] = df.apply(
        lambda row: out_degree[row['from']], axis=1)
    # add in degree  to node
    df['to_in_degree'] = df.apply(lambda row: in_degree[row['to']], axis=1)
    # add out degree  to node
    df['to_out_degree'] = df.apply(lambda row: out_degree[row['to']], axis=1)

    # get new cols
    new_cols = ['from_in_degree', 'from_out_degree',
                'to_in_degree', 'to_out_degree']

    return new_cols


# put all enrich jobs in a dict
dict_enrich = {
    'add_is_holiday': add_is_holiday,
    'add_distance_to_next_holiday': add_distance_to_next_holiday,
    'add_distance_to_previous_holiday': add_distance_to_previous_holiday,
    'add_distance_to_holidays': add_distance_to_holidays,
    'add_day_of_year': add_day_of_year,
    'encode_locations': encode_loc,
    'add_path_distance': add_path_distance,
    'add_path_embedding': add_path_embedding,
    'special_loc_encoding': special_one_hot_encode
}
