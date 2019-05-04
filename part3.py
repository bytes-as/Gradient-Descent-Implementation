# ## 3. Experimenting with larger training set
# ###### [10 marks]
# Repeat the above experiment with three other datasets having size 100, 1000 and 10,000 instances (each dataset generated similarly as described in Part 1a).
# Draw the learning curve of how train and test error varies with increase in size of datasets (for 10, 100, 1000 and 10000 instances).

# In[633]:


import random; # to genearte the uniform number in closed range
import math;   # to find the sin value
import numpy;  # to store the data
import pandas;
import matplotlib.pyplot as plt # for plotting
import unicodedata
best_n = 6;


# In[634]:


#  define the hypothesis function
def hFunction(x, n, coef):
    y = 0.0;
    for i in range(n+1):
#         print(i);
        y += coef[i] * x**i;
    return y;

# Function is as follows :
# F(X) = coef[0] + coef[1]*X^1 + coef[2]*X^^2 + ...
# in assignment coef is reffered as weight


# In[635]:


def costFunction(dataset, weightT, n, number_of_sample):
    result = 0.0;
    for i in range(number_of_sample):
        result += (hFunction(dataset.iloc[i]['X'], n, weightT) - dataset.iloc[i]['Y'])**2;
    result /= 2*number_of_sample;
    return result;

def costFunctionDerivative(dataset, weightT, n, number_of_sample, respect_to_i):
    result = 0.0;
    for i in range(number_of_sample):
        result += (hFunction(dataset.iloc[i]['X'], n, weightT) - dataset.iloc[i]['Y']) * dataset.iloc[i]['X']**respect_to_i;
    result /= number_of_sample;
    return result;


# In[636]:


def meanSquareError(dataset, number_of_entries, weight, n):
    sq_err = 0.0;
    for i in range(0, number_of_entries):
        sq_err += (hFunction(dataset.iloc[i]['X'], n, weight) - dataset.iloc[i]['Y'])**2;
    sq_err = sq_err / (2*(number_of_entries-1));
    return sq_err;


# In[637]:


def gradDesc(n, training_set, number_of_training_samples, maximum_number_of_iteration):
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
        new_error = costFunction(training_set, weight, n, number_of_training_samples);
        if old_error - new_error < 0.00001:
            print("Error isn't decreasing more than 0.000001 in one step anymore.");
            break;
        old_error = new_error;
        i += 1;
        if i > maximum_number_of_iteration:
            print("Number of steps crossed over " + str(maximum_number_of_iteration) + ".");
            break;
        for j in range(n+1):
            deriv = costFunctionDerivative(training_set, old_weight, n, number_of_training_samples, j);
            weight[j] = old_weight[j] - 0.05 * deriv;
        for j in range(n+1):
            old_weight[j] = weight[j];
#         print("i = " + str(i) + " | error = " +str(squareMeanError(training_set, number_of_terms, weight)));
    return weight;
# print(gradDesc(1, training_set, 8, 600));


# In[640]:


number = [10, 100, 1000, 10000];
train_errors = numpy.empty([0,1]);
test_errors = numpy.empty([0,1]);
weights = numpy.empty([0,best_n+1]);
for number in [10, 100, 1000, 10000]:
    arr = numpy.empty([0,2]);
    for i in range(number):
        x = random.uniform(0, 1);
        y = math.sin(2*math.pi*x) + numpy.random.normal(0, 0.3);
        arr = numpy.append(arr, [[x,y]], axis=0);
    random_dataset = pandas.DataFrame(data=arr, columns=['X', 'Y']);
    random_dataset.to_csv("random_dataset_part3_"+str(number)+".csv", sep=",", columns=['X', 'Y']);

    #plotting the sin2pix curve
    x = numpy.linspace(-0.000001, 1.0000001, 100000);
    y = [];
    for i in range(100000):
        y.append(math.sin(2* math.pi * x[i]));
    plt.plot(x, y);

    numpy.random.shuffle(arr);

    training_set = numpy.empty([0,2]);
    for i in range((int)(0.8*number)):
        training_set = numpy.append(training_set, [arr[i]], axis = 0);
    training_set = pandas.DataFrame(data=training_set, columns=['X', 'Y']);
    plt.scatter(training_set[:]['X'], training_set[:]['Y'], color='r', s=1);
    training_set.to_csv("training_set_Part3_" + str(number) + ".csv", sep=",", index=False);

    test_set = numpy.empty([0,2]);
    for i in range((int)(0.8*number), number):
        test_set = numpy.append(test_set, [arr[i]], axis=0);
    test_set = pandas.DataFrame(data=test_set, columns=['X', 'Y']);
    plt.scatter(test_set[:]['X'], test_set[:]['Y'], color='g', s=1);
    plt.savefig("number_of_sample_" + str(number) + ".jpeg");
    test_set.to_csv("test_set_part3_" + str(number) + ".csv", sep=",", index=False);
    plt.show();
    weight = gradDesc(best_n, training_set, (int)(0.8*number), 1000);
    print(weight);
    weights = numpy.append(weights, [weight], axis=0);
    #################### got the weight ##########################
    #################### need to find the errors #################
    ##############################################################
    train_error = meanSquareError(training_set, (int)(0.8 * number), weight, best_n);
    test_error  = meanSquareError(test_set, (int)(0.2 * number), weight, best_n);
    train_errors = numpy.append(train_errors, [[train_error]], axis=0);
    test_errors = numpy.append(test_errors, [[test_error]], axis=0);
# print(train_errors);
# print(test_errors);
# print(weights);
weights = pandas.DataFrame(data=weights);
column_name = [];
for i in range(best_n+1):
        column_name.append('W'+str(i));
weights.columns = column_name;

train_errors = pandas.DataFrame(data=train_errors, columns=['train_error']);
test_errors = pandas.DataFrame(data=test_errors, columns=['test_error']);
number = pandas.DataFrame({'number_of_sample':[10, 100, 1000, 10000]});
errors = train_errors.join(test_errors);
# print(errors);
# print(number);
number = number.join(errors);
# print(number);
# print(weights);
number = number.join(weights);
# print(number);
final_result = number;
print(number);
final_result.to_csv("result_part3.csv", sep=",", index=False);
plt.plot(number.iloc[:]['number_of_sample'], number.iloc[:]['train_error'], 'rs');
plt.plot(number.iloc[:]['number_of_sample'], number.iloc[:]['test_error'], 'g^');
plt.savefig("train_test_error_large_dataset.jpeg");
plt.show();
