# K-Nearest Neighbour on handwritten numbers

Configured to: perform KNN with a k from 1 to 99, and k cross validation with a cv of 10, this translates to a split of 90/10. This can all be adjusted in the `run.py` script, from which the framework is also executed.

*Note: that you must first unpack the datasets located in `/data/`.*

## Output
The framework outputs the `fitting time`, `test set computation time`, `training set computation time`, `training set results` and `test set results` together with the mean and standard deviation of each.

Furthermore, it outputs the mean and standard deviation of the computation time and accuracy of the results obtained from k-cross validation.

## Features

* csv loading
* cross validation
* knn
* performance evaluatuon
* logging
* data shuffling
* plotting

### Dependencies

* python 3.7.4
* sklean
* matplotlib
* pandas
* numpy

### Authors

niste15@student.sdu.dk <br>
moole15@student.sdu.dk <br>
frsoe14@student.sdu.dk <br>
emstu15@student.sdu.dk