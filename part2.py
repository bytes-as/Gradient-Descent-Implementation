# ## 2. Visualization of the dataset and the fitted curves
# ###### [10 + 10 = 20 marks]
#     (a). Draw separate plots of the synthetic data points generated in 1 (a), and all 9 different curves that you have fit for the given dataset in 1 (c).

# In[656]:


import numpy;
import matplotlib.pyplot as plt;
import pandas;
import math;


# In[657]:


# hypothesis function
def hFunction(x, number_of_terms, coef):
    y = 0.0;
    for i in range (number_of_terms):
#         print(i);
        y += coef[i] * x**i;
    return y;
# Function is as follows :
# F(X) = coef[0] + coef[1]*X^1 + coef[2]*X^^2 + ...
# in assignment coef is reffered as weight


# In[658]:


training_set = pandas.read_csv("training_set.csv");
print("Training SET :");
print(training_set);
test_set = pandas.read_csv("test_set.csv");
print("\nTest_Set : ");
print(test_set);


# In[659]:


sin_x = numpy.linspace(-0.2,1.2,1000);
sin_y = numpy.empty([0, 1]);
for i in range(1000):
    temp = math.sin(math.pi * 2 * sin_x[i]);
    sin_y = numpy.append(sin_y, [[temp]]);
# print(sin_y);
plt.plot(sin_x, sin_y);


# In[660]:


for i in range(2,11):
    weightT = numpy.genfromtxt("weight_"+str(i-1)+".csv", delimiter=",", skip_header=1)
    print ("for n = " + str(i-1));
    print(weightT);
    values = []
    x = numpy.linspace(-0.2, 1.2, 1000);
    for j in range(1000):
        y = hFunction(x[j], i, weightT);
#         print("X = " + str(x[j]) + " | Y = " + str(y));
        values = numpy.append(values, [y], axis=0);
    plt.plot(x,values);
#     plt.plot(sin_x, sin_y, color='g');
    plt.scatter(training_set[:]['X'], training_set[:]['Y'], color='r');
    plt.savefig("fitted_curve_" + str(i) + ".jpeg");
    plt.show();


#         (b). Report squared error on both train and test data for each value of n in the form of a plot where along x-axis, vary n from 1 to 9 and along y-axis, plot both train error and test error. Explain which value of n is suitable for the synthetic dataset that you have generated and why.

# In[661]:


#  define the hypothesis function
# nuber of terms = n+1
def hFunction(x, number_of_terms, coef):
    y = 0.0;
    for i in range (number_of_terms):
#         print(i);
        y += coef[i] * x**i;
    return y;
# Function is as follows :
# F(X) = coef[0] + coef[1]*X^1 + coef[2]*X^^2 + ...
# in assignment coef is reffered as weight


# In[662]:


def meanSquaredError(dataset, number_of_entries, weight, n):
    sq_err = 0.0;
    for i in range(0, number_of_entries):
        sq_err += (hFunction(dataset.iloc[i]['X'], n+1, weight) - dataset.iloc[i]['Y'])**2;
    sq_err = sq_err / (2*(number_of_entries-1));
    return sq_err;


# In[663]:


# updating training_set.csv adding error column
training_error_np = [];
index = []
for i in range(0,9):
    weightT = numpy.genfromtxt("weight_"+str(i+1)+".csv", delimiter=",", skip_header=1)
#     print(weightT);
    training_error = meanSquaredError(training_set, 7, weightT, i+1);
    training_error_np = numpy.append(training_error_np, training_error);
    index = numpy.append(index, i+1);
# print(training_error_np);
train_error = pandas.DataFrame(data=training_error_np, columns=['train_error']);
index = pandas.DataFrame(data=index, columns=['n']);
train_error = index.join(train_error);
# print(train_error);
# train_error.to_csv("train_error.csv", sep=',', index=False);


# In[664]:


test_error_np = [];
index = []
for i in range(0,9):
    weightT = numpy.genfromtxt("weight_"+str(i+1)+".csv", delimiter=",", skip_header=1)
#     print(weightT);
    test_error = meanSquaredError(test_set, 2, weightT, i+1);
    test_error_np = numpy.append(test_error_np, test_error);
    index = numpy.append(index, i+1);
    index = pandas.DataFrame(data=index, columns=['n']);
# print(test_error_np);
test_error = pandas.DataFrame(data=test_error_np, columns=['test_error']);
test_error = train_error.join(test_error);
# print(test_error);


# In[667]:


error_n = test_error;#training_error.join(test_error);
print(error_n);
error_n.to_csv("error_n.csv", sep=',', index=False);
train_plot = plt.plot(train_error[:]['n'], train_error[:]['train_error'], 'rs', test_error[:]['n'], test_error[:]['test_error'], 'g^');
plt.savefig("train_test_error.jpeg")
