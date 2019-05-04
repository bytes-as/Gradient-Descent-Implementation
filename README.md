# Implementation of Gradient Descent with Native Python
Feb 15​ th​ , 2019
### OVERVIEW
This whold project is to generate a dataset over a function following sin(2*pi*x) and add noise
over it following the gaussian distribution, then to fit a curve with the help of Gradient Descent
with different scenarios varying the number of data set samples and with the different degree of
the hypothesis function.

### GOALS : 
  1. Synthetic data generation and simple curve fitting
  2. Visualization of the dataset and the fitted curves
  3. Experimenting with larger training set
  4. Experimenting with cost functions

### How to go about the source code :
There are different scripts each for the goals written above, each file is to be interpreted by
python version : Python 3.6.5 :: Anaconda, Inc. The script for each part will gonna create the files
which contains the data points taken into the processing and finding the parameter/weight
associated with the best fitting curve to that data set taken into the consideration. Along with that
each script file will gonna create some plots referencing some concerning some result
parameters or factors.

#### Goal 1 : __Synthetic data generation and simple curve fitting__
`part1.py` when interpreted with the python it will gonna generate a random dataset
following the sinusoidal function as mentioned in the PS, along with the Gaussian Distribution
following noise. That data set will gonna save in the file ‘random_dataset.csv’
Then it will gonna take random 80% of that random data and this 80% will gonna taken as
training dataset for the gradient algorithm. While the other random 20% will gonna be taken for
the test dataset.Finally the gradient descent is applied over that training dataset with different number of degrees
of the hypothesis function varying from 1 to 9. The weights of all those hypothessis function
haveing degree from 1 to 9 is written in the files `weight_n.csv` where n is the degree of the
hypothesis function.

#### Goal 2 : __Visualization of the dataset and fitted curve__
`part2.py` when interpreted with the python it will gonna generate the plot of the
dataset shoing training data points in red color and test set data points in green color, along with
that it generates all the plot having name "fitted_curve_n.jpeg" which contains the fitted curve
against the the dataset that we have taken into consideration to fit the curve on. And the last plot
will be the train error and test error against the degree of the hypothessis function varting from 1
to 9, named “train_test_error.jpeg”

#### Goal 3 : __Experimenting with larger training set__
`part3.py` plot of the random dataset is saved as the csv file named
random_dataset_part3_n.csv where the n is the number of the datapoints.
And the plot are saved in the file named as number_of_sample_n.jpeg where the n is the number
of dtapoints.
