## NNights-code

Code developed for the **Flight challenge**.

The goal is to predict the number of passengers per plane on some flights in the US.


## Data set

The dataset has the following columns:

- *flight_date* : the date of the flight
- *from*: the code of the departure airport
- *to*: the code of the arrival airport
- *avg_weeks*: the average number of weeks between ticket purchase and flight date
- *std_weeks*: the standard deviation of the number of weeks between ticket purchase and flight date
- *target*: the variable to predict. It relates to the number of passengers on the flight. For privacy reasons, the number of passengers is not available, and the ``target`` variable is available instead.



## Note on scoring

The performance of the prediction will be quantified on left-out data, using the **RMSE** (Root Mean Squared Error).
The test data is available under the same format as the training data, minus the target column.