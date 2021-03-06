# noqa
from typing import List


DICT_ENCODINGS = {'ATL': 0, 'BOS': 1, 'CLT': 2,
                  'DEN': 3, 'DFW': 4, 'DTW': 5,
                  'EWR': 6, 'IAH': 7, 'JFK': 8,
                  'LAS': 9, 'LAX': 10, 'LGA': 11,
                  'MCO': 12, 'MIA': 13, 'MSP': 14,
                  'ORD': 15, 'PHL': 16, 'PHX': 17,
                  'SEA': 18, 'SFO': 19}


def encode_location(arr: List[str]) -> List[int]:
    """Perform Label encoding.

    Parameters
    ----------
    arr : List[str]
        List of locations.

    Returns
    -------
    List[int]
        returns list of the index of locations.
    """
    return [DICT_ENCODINGS[el] for el in arr]
