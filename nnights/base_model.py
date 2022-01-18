"""Base model.

Given source and target, the model predict the mean in this path.
"""

import pandas as pd


class BaseModel:  # noqa
    def __init__(self, cols_path='from to', col_y='target') -> None:  # noqa
        # params
        self.cols_path = cols_path.split(" ")
        self.col_y = col_y
        return

    def fit(self, X: pd.DataFrame):
        """Fit the model given the data.

        Parameters
        ----------
        X : pd.DataFrame
            The flight dataset.
        """
        # params
        cols_path = self.cols_path
        col_y = self.col_y

        # build look up table
        self.table_path_target = X.groupby(by=cols_path).mean()[col_y]
        self.mean_target = self.table_path_target.mean()
        return

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict target given data.

        Parameters
        ----------
        X : pd.DataFrame
            The flight dataset, either train or test

        Returns
        -------
        pd.Series
            returns a serie of predict values.
        """
        # build col (from, to)
        path_col = X["from"] + "," + X["to"]

        return path_col.apply(self._predictor)

    def _predictor(self, tulpe_as_str: str) -> float:
        """Predict a single value given a path (source, destination).

        Parameters
        ----------
        tulpe_as_str : str
            path expressed as string of the form `source,destination`.

        Returns
        -------
        float
            return the target prediction of the path.
        """
        # form el of tuple
        source, destination = [el for el in tulpe_as_str.split(",")]

        # case path exists in table
        try:
            return self.table_path_target.loc[(source, destination)]
        # case path not in table
        except:  # noqa
            return self.mean_target
