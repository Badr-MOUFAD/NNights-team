"""Utils to manipulate holidays.

Utils to extract useful informations from a given
data frame of holidays. Note that the data frame of holidays
must be sorted in an ascendant order.
"""

import pandas as pd


# load usa holidays
USA_HOLIDAYS = pd.read_csv("./data/usa_holidays.csv")


def is_holiday(d: pd.Timestamp) -> bool:
    """Determine if a day is holiday.

    Parameters
    ----------
    d : pd.Timestamp
        date

    Returns
    -------
    bool
        returns whether the current date is a holiday
    """
    # select range of search
    df_potential_holidays = USA_HOLIDAYS.query(f"month == {d.month}")

    # extract day
    day = d.day

    for i in df_potential_holidays.index:
        # current holiday bounds
        start = df_potential_holidays.loc[i, "start"]
        end = df_potential_holidays.loc[i, "end"]

        if start <= day <= end:
            return True

    return False


def distance_next_holiday(d: pd.Timestamp) -> int:
    """Compute the number of days to the next holiday.

    Parameters
    ----------
    d : pd.Timestamp
        date

    Returns
    -------
    int
        returns the number of days

    """
    # case d > latest(holidays)
    current_year = d.year

    # select latest holiday
    last_index = len(USA_HOLIDAYS) - 1
    latest_holiday = {
        "day": USA_HOLIDAYS.loc[last_index, "start"],
        "month": USA_HOLIDAYS.loc[last_index, "month"],
        "year": current_year
    }

    if pd.Timestamp(**latest_holiday) < d:
        first_holiday = {
            "day": USA_HOLIDAYS.loc[0, "start"],
            "month": USA_HOLIDAYS.loc[0, "month"],
            "year": current_year + 1
        }
        duration = pd.Timestamp(**first_holiday) - d
        return duration.days

    # select area of search
    df_selected_holidays = USA_HOLIDAYS.query(f"month >= {d.month}")

    for i in df_selected_holidays.index:
        current_holiday = {
            "day": df_selected_holidays.loc[i, "start"],
            "month": df_selected_holidays.loc[i, "month"],
            "year": current_year
        }

        duration = pd.Timestamp(**current_holiday) - d

        if duration.days >= 0:
            return duration.days


def distance_previous_holiday(d: pd.Timestamp) -> int:
    """Compute the number of days to the previous holiday.

    Parameters
    ----------
    d : pd.Timestamp
        date

    Returns
    -------
    int
        returns the number of days

    """
    # case d < first(holidays)
    current_year = d.year

    # select first holiday
    first_holiday = {
        "day": USA_HOLIDAYS.loc[0, "end"],
        "month": USA_HOLIDAYS.loc[0, "month"],
        "year": current_year
    }

    if d < pd.Timestamp(**first_holiday):
        latest_index = len(USA_HOLIDAYS) - 1

        # select latest holiday
        latest_holiday = {
            "day": USA_HOLIDAYS.loc[latest_index, "end"],
            "month": USA_HOLIDAYS.loc[latest_index, "month"],
            "year": current_year - 1
        }

        duration = d - pd.Timestamp(**latest_holiday)
        return duration.days

    # select area of search
    df_selected_holidays = USA_HOLIDAYS.query(f"month <= {d.month}")

    for i in df_selected_holidays.index[::-1]:
        current_holiday = {
            "day": df_selected_holidays.loc[i, "end"],
            "month": df_selected_holidays.loc[i, "month"],
            "year": current_year
        }

        duration = d - pd.Timestamp(**current_holiday)

        if duration.days >= 0:
            return duration.days


def distance_to_holidays(d: pd.Timestamp) -> pd.Series:
    """Compute distance between date and all holidays.

    We use `min(abs(.))` as a norm to compare distances.

    Parameters
    ----------
    d : pd.Timestamp
        date

    Returns
    -------
    pd.Series
        returns a Serie of distances between the given day and holidays

    """
    year = d.year

    # where to store distances
    prev_current_next_distances = {
        "prev": [],
        "current": [],
        "next": []
    }

    # compute distances
    for i in USA_HOLIDAYS.index:
        # holiday infos
        month = USA_HOLIDAYS.loc[i, "month"]
        duration = USA_HOLIDAYS.loc[i, "duration"]
        day = USA_HOLIDAYS.loc[i, "start"] + int(duration / 2)

        for incr, name_year in zip([-1, 0, 1], prev_current_next_distances):
            holiday_date = pd.Timestamp(f"{year+incr}-{month}-{day}")
            distance = holiday_date - d

            prev_current_next_distances[name_year].append(distance.days)

    # select relevant distances
    df_distances = pd.DataFrame(
        prev_current_next_distances,
        index=USA_HOLIDAYS["name"]
    )

    # func to use in aggregation
    def agg_function(arr):
        return min(arr, key=lambda x: abs(x))

    return df_distances.agg(agg_function, axis=1)
