# ## 4. Experimenting with cost functions
# ###### [20 + 10 = 30 marks]
#     (a). Solve the problem by minimizing different cost functions (Do not use any regularization, Use gradient descent to minimize the cost function in each case) :
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i). Mean absolute error i.e.                               $J (θ) = 1/2m(\sum\limits_{i = 1}^{m}| W^T Φ n (x) − y |)$<br>
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(ii). Fourth power error i.e.                              $J (θ) = 1/2m(\sum\limits_{i = 1}^{m}| W^T Φ n (x) − y |^4)$<br>
#
#     (b). Plot the test RMSE vs learning rate for each of the cost functions. Vary the learning rates as 0.025, 0.05, 0.1, 0.2 and 0.5. Which one would you prefer for this problem and why?

# In[639]:


import random; # to genearte the uniform number in closed range
import math;   # to find the sin value
import numpy;  # to store the data
import pandas;
import matplotlib.pyplot as plt # for plotting
import unicodedata


# In[537]:


# calling uniform but that is same as random because of the range is 0 to 1
arr = numpy.empty([0, 2]);
for i in range(50):
    x = random.uniform(0, 1);
    y = math.sin(2*math.pi*x) + numpy.random.normal(math.sin(2*math.pi*x), 0.3);
#     print("x = " + str(x) + " | y = " + str(y));
#     taking mean 0.5 in above expression to generate the noise following gaussian distribution
    arr = numpy.append(arr, [[x,y]], axis=0);


# In[538]:


random_dataset = pandas.DataFrame(data=arr, columns=['X', 'Y']);
random_dataset.to_csv("random_dataset.csv", sep=",", index=False);
print("data has been generated successfully");
print("Dataset is as follows : ")
print(random_dataset);
random_dataset_plot = plt.scatter(random_dataset[:]['X'], random_dataset[:]['Y'], color='r', s=1);
# print(type(random));


# In[539]:


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


# In[540]:


numpy.random.shuffle(arr);


# In[541]:


train_set = numpy.empty([0, 2]);
for i in range(0,40):
    train_set = numpy.append(train_set, [arr[i]], axis=0);
## converting to pandas dataframe
train_set = pandas.DataFrame(data=train_set, columns=['X', 'Y']);
print(train_set);
train_set.to_csv("train_set.csv", ",", index=False);

test_set = numpy.empty([0, 2]);
for i in range(40,50):
    test_set = numpy.append(test_set, [arr[i]], axis=0);
## converting to pandas dataframe
test_set = pandas.DataFrame(data=test_set, columns=['X', 'Y']);
print(test_set);
# test_set.to_csv("test_set.csv", ",", index=False);


# In[566]:


train_set_plot = plt.scatter(train_set[:]['X'], train_set[:]['Y'], color='r');
test_set_plot = plt.scatter(test_set[:]['X'], test_set[:]['Y'], color='g');
plt.plot(sin_x, sin_y);


# In[567]:


#  define the hypothesis function
def hFunction(x, n, coef):
    y = 0.0;
    for i in range (n+1):
#         print(i);
        y += coef[i] * x**i;
    return y;
# Function is as follows :
# F(X) = coef[0] + coef[1]*X^1 + coef[2]*X^^2 + ...
# in assignment coef is reffered as weight


# In[568]:


def costFunction1(training_set, weightT, n, number_of_sample):
    mean_absolute_error = 0.0;
    for i in range(number_of_sample):
        mean_absolute_error += abs(hFunction(training_set.iloc[i]['X'],n, weightT) - training_set.iloc[i]['Y']);
    mean_absolute_error /= 2*number_of_sample;
    return mean_absolute_error;

def costFunction1Derivative(training_set, weightT, n, number_of_training_samples, respect_to):
    result = 0.0;
    for i in range(number_of_training_samples):
        if (hFunction(training_set.iloc[i]['X'], n, weightT) - training_set.iloc[i]['Y']) < 0:
            result += -training_set.iloc[i]['X'];
        else :
            result += training_set.iloc[i]['X'];
    result /= 2*number_of_training_samples;
    return result;


# In[586]:


def costFunction2(training_set, weightT, n, number_of_sample):
    mean_fourth_error = 0.0;
    for i in range(number_of_sample):
        mean_fourth_error = (hFunction(train_set.iloc[i]['X'], n, weightT)-training_set.iloc[i]['Y'])**4;
    mean_fourth_error /= 2*number_of_sample;
    return mean_fourth_error;

def costFunction2Derivative(training_set, weightT, n , number_of_samples, respect_to):
    result = 0.0;
    for i in range(number_of_samples):
        result += ((hFunction(training_set.iloc[i]['X'], n, weightT) - training_set.iloc[i]['Y'])**3) * training_set.iloc[respect_to]['X'];
    result /= number_of_samples/2;
    return result;


# In[587]:


def gradDesc1(n, train_set, number_of_training_samples, maximum_number_of_iteration, learning_rate):
    weight = numpy.empty([0,1]);
    old_weight = numpy.empty([0, 1]);
    # initializing the values of the pre assumed weight of the values of X
    for i in range (n+1):
        weight = numpy.append(weight, [0]);
        old_weight = numpy.append(old_weight, [0]);
    old_error = 10000000;
    new_error = 0;
    i = 0;
    while(1):
#         new_error = squareMeanError(train_set, number_of_terms, weight);
        new_error = costFunction1(train_set, weight, n, number_of_training_samples);
        if old_error - new_error < 0.00001:
            print("Error isn't decreasing more than 0.000001 in one step anymore.");
            break;
        old_error = new_error;
        i += 1;
        if i > maximum_number_of_iteration:
            print("Number of steps crossed over " + str(maximum_number_of_iteration) + ".");
            break;
        for j in range(n+1):
            deriv = costFunction1Derivative(train_set, old_weight, n, number_of_training_samples, j);
            weight[j] = old_weight[j] - learning_rate * deriv;
        for j in range(n+1):
            old_weight[j] = weight[j];
#         print("i = " + str(i) + " | error = " +str(squareMeanError(training_set, number_of_terms, weight)));
    return weight;


# In[588]:


def gradDesc2(n, train_set, number_of_training_samples, maximum_number_of_iteration, learning_rate):
    weight = numpy.empty([0,1]);
    old_weight = numpy.empty([0, 1]);
    # initializing the values of the pre assumed weight of the values of X
    for i in range (n+1):
        weight = numpy.append(weight, [0]);
        old_weight = numpy.append(old_weight, [0]);
    old_error = 10000000;
    new_error = 0;
    i = 0;
    while(1):
#         new_error = squareMeanError(train_set, number_of_terms, weight);
        new_error = costFunction2(train_set, weight, n, number_of_training_samples);
        if old_error - new_error < 0.00001:
            print("Error isn't decreasing more than 0.000001 in one step anymore.");
            break;
        old_error = new_error;
        i += 1;
        if i > maximum_number_of_iteration:
            print("Number of steps crossed over " + str(maximum_number_of_iteration) + ".");
            break;
        for j in range(n+1):
            deriv = costFunction2Derivative(train_set, old_weight, n, number_of_training_samples, j);
            weight[j] = old_weight[j] - learning_rate * deriv;
        for j in range(n+1):
            old_weight[j] = weight[j];
#         print("i = " + str(i) + " | error = " +str(squareMeanError(training_set, number_of_terms, weight)));
    return weight;


# In[589]:


# calling gradiend descent optimisation fot different values of n varying from 1 to 9
# number of terms = n + 1
print ("for n = " + str(best_n));
weightT1 = gradDesc1(best_n, train_set, 40, 1000000, 0.05);
weightT2 = gradDesc2(best_n, train_set, 40, 1000000, 0.05);
print("values of the weights for n = " + str(best_n));
print(weightT1);
print(weightT2);
# with open('weight_'+ str(i-1) + '.csv', mode='w') as abc:
#     abc.write("n = " + str(i-1) + "\n");
#     numpy.savetxt(abc, weightT, delimiter="\n");
# numpy.savetxt(file, weightT, delimiter=',');
# print(weightT);
# values = []
# x = numpy.linspace(-0.2, 1.2, 1000);
# print(type(x));
# for j in range(1000):
#     y = hFunction(x[j], i, weightT);
#         print("X = " + str(x[j]) + " | Y = " + str(y));
#         values = numpy.append(values, [y], axis=0);
#     plt.plot(x,values);
#     plt.scatter(training_set[:]['X'], training_set[:]['Y'], color='r');
#     plt.show();
