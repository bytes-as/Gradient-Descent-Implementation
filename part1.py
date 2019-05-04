
# coding: utf-8

# # Machine Learning -- Linear Regression using Gradient Descent

# ## 1. *Synthetic* data generation and simple curve fitting
# ###### 10 + 5 + 25 = 40 marks
#
# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    a. Generate a synthetic dataset as follows.
# The input values { x i } are generated uniformly in range [ 0, 1] , and the corresponding target values { y i } are obtained by first computing the corresponding values of the function __Sin(2$\pi$x)__ , and then adding a random noise with a __Gaussian distribution__ having *standard deviation* __0.3__. Generate ​ 10 such instances of
# (x i , y i ) . \[You can use any standard module to generate random numbers as per a
# gaussian / normal distribution, e.g., numpy.random.normal for python.\]

# In[642]:




import random; # to genearte the uniform number in closed range
import math;   # to find the sin value
import numpy;  # to store the data
import pandas;
import matplotlib.pyplot as plt # for plotting
import unicodedata


# In[643]:


# calling uniform but that is same as random because of the range is 0 to 1
arr = numpy.empty([0, 2]);
for i in range(10):
    x = random.uniform(0, 1);
    y = math.sin(2*math.pi*x) + numpy.random.normal(0, 0.3);
#     print("x = " + str(x) + " | y = " + str(y));
#     taking mean 0.5 in above expression to generate the noise following gaussian distribution
    arr = numpy.append(arr, [[x,y]], axis=0);
print(arr);


# In[644]:


random_dataset = pandas.DataFrame(data=arr, columns=['X', 'Y']);
random_dataset.to_csv("random_dataset.csv", sep=",", index=False);
print("data has been generated successfully");
print("Dataset is as follows : ")
print(random_dataset);
random_dataset_plot = plt.scatter(random_dataset[:]['X'], random_dataset[:]['Y']);
# print(type(random));


# In[645]:


print("Points are generated from is sin(2" + unicodedata.lookup("GREEK SMALL LETTER PI") + "x): ");
sin_x = numpy.linspace(-0.2,1.2,1000);
sin_y = numpy.empty([0, 1]);
for i in range(1000):
    temp = math.sin(math.pi * 2 * sin_x[i]);
    sin_y = numpy.append(sin_y, [[temp]]);
# print(sin_y);
plt.plot(sin_x, sin_y);
random_dataset_plot = plt.scatter(random_dataset[:]['X'], random_dataset[:]['Y'], color='r');
plt.show();


# ###  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  b. Split the dataset into two sets randomly:
# (i) Training Set (80%) (ii) Test Set (20%).

# In[646]:


# to randomise the data set and test set, appling shuffle to the array
numpy.random.shuffle(arr);
# print("SHUFFELED DATA SET IS AS FOLLOWS : ");
# print(arr);


# In[647]:


# defining the training set
training_set = numpy.empty([0, 2]);
for i in range(0,8):
    training_set = numpy.append(training_set, [arr[i]], axis=0);
## converting to pandas dataframe
training_set = pandas.DataFrame(data=training_set, columns=['X', 'Y']);
print(training_set);
training_set.to_csv("training_set.csv", ",", index=False);


# In[648]:


# defining the test set
test_set = numpy.empty([0, 2]);
for i in range(8, 10):
    test_set = numpy.append(test_set, [arr[i]], axis=0);
test_set = pandas.DataFrame(data=test_set, columns=['X','Y']);
print(test_set);
test_set.to_csv("test_set.csv", ",", index=False);


# In[650]:


training_set_plot = plt.scatter(training_set[:]['X'], training_set[:]['Y'], color='r');
test_set_plot = plt.scatter(test_set[:]['X'], test_set[:]['Y'], color='g');
plt.plot(sin_x, sin_y);
plt.savefig("random_datset.jpeg");
plt.show();


# ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;C. Write a code to fit a curve that minimizes ​ squared error cost function ​ using gradient descent (with learning rate 0.05):
# As discussed in class, on the training set while the model takes following form $y = W^T Φ_n (x)$ , W ∈ R n+1 ,$Φ_n (x) = [1, x, x^2 , x^3 ... , x^n ] $. Squared error is defined as $J (θ) = \frac{1}{2m}(\sum\limits_{i = 1}^{m}W^T Φ_n(x) − y_i)$. In your experiment, vary n from 1 to 9. In other words, fit 9 different curves to the training data, and hence estimate
# the parameters. Use the estimated W to measure squared error on the test set, and name it as test error on test data.

# In[651]:


#  define the hypothesis function
def hFunction(x, number_of_terms, coef):
    y = 0.0;
    for i in range (number_of_terms):
#         print(i);
        y += coef[i] * x**i;
    return y;
# Function is as follows :
# F(X) = coef[0] + coef[1]*X^1 + coef[2]*X^^2 + ...
# in assignment coef is reffered as weight


# In[652]:


def costFunction(x, y, weightT, number_of_terms):
    return(hFunction(x, number_of_terms, weightT) - y);

def costFunctionDerivative(training_set, weightT, number_of_terms, number_of_training_samples, respect_to):
    result = 0.0;
    for i in range(number_of_training_samples):
        result += 2 * (hFunction(training_set.iloc[i]['X'], number_of_terms, weightT) - training_set.iloc[i]['Y']) * training_set.iloc[i]['X']**respect_to;
    result /= 2*number_of_training_samples;
    return result;


# In[653]:


# return squared error of the training data set
def squareMeanError(training_set, number_of_terms, weightT):
    error = 0.0;
    for i in range(8):
        error += (costFunction(training_set.iloc[i]['X'], training_set.iloc[i]['Y'], weightT, number_of_terms))**2;
    error /= 2*8;
    return error;

# weightT = [1, 1];
# error = squareMeanError(training_set, 2, weightT);
# print("error = " + str(error));


# In[654]:


def gradDesc(number_of_terms, training_set, number_of_training_samples, maximum_number_of_iteration):
    weight = numpy.empty([0,1]);
    old_weight = numpy.empty([0, 1]);
    # initializing the values of the pre assumed weight of the values of X
    for i in range (number_of_terms):
        weight = numpy.append(weight, [5]);
        old_weight = numpy.append(old_weight, [5]);
    old_error = 10000000;
    new_error = 0;
    i = 0;
    while(1):
        new_error = squareMeanError(training_set, number_of_terms, weight);
        if old_error - new_error < 0.00001:
            print("Error isn't decreasing more than 0.000001 in one step anymore.");
            break;
        old_error = new_error;
        i += 1;
        if i > maximum_number_of_iteration:
            print("Number of steps crossed over " + str(maximum_number_of_iteration) + ".");
            break;
        for j in range(number_of_terms):
            deriv = costFunctionDerivative(training_set, old_weight, number_of_terms, number_of_training_samples, j);
            weight[j] = old_weight[j] - 0.05 * deriv;
        for j in range(number_of_terms):
            old_weight[j] = weight[j];
#         print("i = " + str(i) + " | error = " +str(squareMeanError(training_set, number_of_terms, weight)));
    return weight;
# print(gradDesc(1, training_set, 8, 600));


# In[655]:


# calling gradiend descent optimisation fot different values of n varying from 1 to 9
# number of terms = n + 1
for i in range(2,11):
    print ("for n = " + str(i-1));
    weightT = gradDesc(i, training_set, 8, 1000);
    print("values of the weights for n = " + str(i-1));
    with open('weight_'+ str(i-1) + '.csv', mode='w') as abc:
        abc.write("n = " + str(i-1) + "\n");
        numpy.savetxt(abc, weightT, delimiter="\n");
#     numpy.savetxt(file, weightT, delimiter=',');
    print(weightT);
    values = []
    x = numpy.linspace(-0.2, 1.2, 1000);
#     print(type(x));
    for j in range(1000):
        y = hFunction(x[j], i, weightT);
#         print("X = " + str(x[j]) + " | Y = " + str(y));
        values = numpy.append(values, [y], axis=0);
    plt.plot(x,values);
    plt.scatter(training_set[:]['X'], training_set[:]['Y'], color='r');
    plt.savefig("fitted_curve_" + str(i) + ".jpeg");
    plt.show();
