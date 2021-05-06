## Table of Contents
[Lab 1](https://kaitlinnnn.github.io/DATA310/#lab-1)  

[Lab 2](https://kaitlinnnn.github.io/DATA310/#lab-2)  

[Lab 3](https://kaitlinnnn.github.io/DATA310/#lab-3)  

[Lab 4](https://kaitlinnnn.github.io/DATA310/#lab-4)  

[Midterm Project](https://kaitlinnnn.github.io/DATA310/#midterm-project)  

[Lab 5](https://kaitlinnnn.github.io/DATA310/#lab-5)  

[Lab 6](https://kaitlinnnn.github.io/DATA310/#lab-6)  

[Lab 7](https://kaitlinnnn.github.io/DATA310/#lab-7)  


## Lab 1
(1). What would be the most commonly used level of measurement if the variable is the temperature of the air?

         interval
         
(2). Write a Python code to import the data file 'L1data.csv' (introduced in Lecture 1) and code an imputation for replacing the NaN values in the "Age" column   with the median of the column. The NaN instances are replaced by:
```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = pd.read_csv("L1Data.csv")

imputing_configuration = SimpleImputer(missing_values = np.nan, strategy = 'median')
imp = imputing_configuration.fit(data[["Age"]])

# The following replaces the NaNs in the "Age" column with the median value
data[["Age"]] = imp.transform(data[["Age"]]).ravel()
data.head(13)
```
answer:
```
21.0
```
(3). In Bayesian inference the "likelihood" represents:

         How probable is the data (evidence) given that our hypothesis is true.
                  
(4). The main goal of Monte Carlo simulations is to solve problems by approximating a probability value via carefully designed simulations.

         true
         
(5). Assume that during a pandemic 15% of the population gets infected with a respiratory virus while about 35% of the population has some general respiratory symptoms such as sneezing, stuffy nose etc. Assume that approximately 30% of the people infected with the virus are asymptomatic. What is the probability that someone who has the symptom actually has the disease?
```
P(A | B) = ( P( B | A) * P(A) ) / P(B)

P(A) = symptoms = .35
P(B) = infected = .15
P(B | A) = infected -> symptomatic  = .70

P(A | B) = symptoms -> infected ( .70 * .15 ) / .35 = 0.3
```     
answer:
```
30%
```

(6). A Monte Carlo simulation should never include more than 1000 repetitions of the experiment.

         false
         
(7). One can decide that the number of iterations in a Monte Carlo simulation was sufficient by visualizing a Probability-Iteration plot and determining where the probability graph approaches a horizontal line.

         true
         
(8). Assume we play a slightly bit different version of the original Monte Hall problem such as having four doors one car and three goats. The rules of the game are the same, the contestant chooses one door (that remains closed) and one of the other doors who had a goat behind it is being opened. The contestant has to make a choice as to stick with the original choice or rather switch for one of the remaining closed doors. Write a Python code to approximate the winning probabilities, for each choice, by the means of Monte Carlo simulations. The probability that the contestant will ultimately win by sticking with the original choice is closer to:

```
 You Pick  |  Prize Door  |  Dont Switch  |  Switch
    1              1             Win          Lose
    1              2             Lose         Win
    1              3             Lose         Win
    1              4             Lose         Win
    2              1             Lose         Win
    2              2             Win          Lose
    2              3             Lose         Win
    2              4             Lose         Win
    3              1             Lose         Win
    3              2             Lose         Win
    3              3             Win          Lose
    3              4             Lose         Win
    4              1             Lose         Win
    4              2             Lose         Win
    4              3             Lose         Win
    4              4             Win          Lose
```
answer:
```
5/16 scenarios are loses when the contestant switches doors.
5/16 = 0.3125
= 25%
```
         
(9). In Python one of the libraries that we can use for generating repeated experiments in Monte Carlo simulations is:

         random
         
(10). In Python, for creating a random permutation of an array whose entries are nominal variables we used:

         random.shuffle
         

## Lab 2
(1).
(2).
(3).
(4).
(5).
(6).
(7).
(8).
(9).
(10).

## Lab 3

(1). An "ordinary least squares" (or OLS) model seeks to minimize the differences between your true and estimated dependent variable.

         True



(2). Do you agree or disagree with the following statement: In a linear regression model, all feature must correlate with the noise in order to obtain a good fit.

         Disagree, the noise refers to features that do no correlate well, so less noise means there is a better fit.

   

(3). Write your own code to import L3Data.csv into python as a data frame. Then save the feature values 'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234. If we use the features of x to build a multiple linear regression model for predicting y then the root mean square error on the test data is close to:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv("L3Data.csv")
del df["questions"]

y = df['Grade'].values
X = df.loc[ : , (df.columns != 'Grade') ].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

pred = lin_reg.predict(X_test)

print(metrics.mean_squared_error(y_test, pred)) 
```

```
Output: 69.29694824390998
Final Answer: 8.3244
```


(4). In practice we determine the weights for linear regression with the "X_test" data.

         False.


(5). Polynomial regression is best suited for functional relationships that are non-linear in weights.

         False


(6). Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python.

         True



(7). Write your own code to import L3Data.csv into python as a data frame. Then save the feature values 'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("L3Data.csv")
del df["questions"]

y = df['Grade'].values
X = df.loc[ : , (df.columns != 'Grade') ].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

print(len(X_train))
```

```
Output: 23
```

(8). The gradient descent method does not need any hyperparameters.

         False


(9). To create and display a figure using matplotlib.pyplot that has visual elements (scatterplot, labeling of the axes, display of grid), in what order would the below code need to be executed?

```python
# 1
import matplotlib.pyplot as plt

# 2
fig, ax = plt.subplots()

# 3
ax.scatter(X_test, y_test, color="black", label="Truth")
ax.scatter(X_test, lin_reg.predict(X_test), color="green", label="Linear")
ax.set_xlabel("Discussion Contributions")
ax.set_ylabel("Grade")

# 4
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
```


(10). Which of the following forms is *not linear in the weights* ?

$$

B_1\begin{vmatrix}
\mathbf{x_{11}} \\
\mathbf{x_{21}} \\
\mathbf{x_{31}} \\
\end{vmatrix} +
{e}^{B_1^2+1}\begin{vmatrix}
\mathbf{x^2_{11}} \\
\mathbf{x^2_{21}} \\
\mathbf{x^2_{31}} \\
\end{vmatrix} +
(B_1)^4\begin{vmatrix}
\mathbf{x_{12}} \\
\mathbf{x_{22}} \\
\mathbf{x_{32}} \\
\end{vmatrix}

$$

                  The option with the "e" and the exponents preceding the matrices is not linear.

## Lab 4

(1). Regularization is defined as

         the minimization of the sum of squared residuals subject to a constraint on the weights (aka coefficients)

(2). The regularization with the square of an L2 distance may improve the results compared to OLS when the number of features is higher than the number of observations.

         true

(3). The L1 norm always yields shorter distances compared to the Euclidean norm.

         false

(4). Typically, the regularization is achieved by

         minimizing the sum of squared residuals times the average of the L1 norm of the coefficients.

(5). A regularization method that facilitates variable selection (estimating some coefficients as zero) is

         lasso

(6). Write your own Python code to import the Boston housing data set (from the sklearn library) and scale the data (not the target) by z-scores. If we use all the features with the Linear Regression to predict the target variable then the root mean squared error (RMSE) is:
```python
import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

ss = StandardScaler()
xs = pd.DataFrame(ss.fit_transform(df), columns=df.columns)

lm = LinearRegression()
lm.fit(xs, y)
yhat = lm.predict(xs)

print(math.sqrt(MSE(y, yhat)))
```
output:
```
4.679191295697282
```
(7). On the Boston housing data set if we consider the Lasso model with 'alpha=0.03' then the 10-fold cross-validated prediction error is: (for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)
```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso as lso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target
x = data.data

lasso = lso(alpha=0.03)

kf = KFold(n_splits=10, random_state=1234, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
Output:
```
4.786745365806388
```

(8). On the Boston housing data set if we consider the Elastic Net model with 'alpha=0.05' and 'l1_ratio=0.9' then the 10-fold cross-validated prediction error is: (for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)
```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso as lso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target
x = data.data

lasso = lso(alpha=0.03)

kf = KFold(n_splits=10, random_state=1234, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
output:
```
4.785491295697282
```

(9). If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply OLS, the root mean squared error is:
```python
import numpy as np
import pandas as pd
import math
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

ss = StandardScaler()
xs = pd.DataFrame(ss.fit_transform(df), columns=df.columns)

pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(xs)

model = LinearRegression()
model.fit(xp, y)
ypp = model.predict(xp)

rmse = np.sqrt(MSE(y,ypp))
print(rmse)
```
output:
```
2.448373257727784
```

(10). If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply the Ridge regression with alpha=0.1 and we create a Quantile-Quantile plot for the residuals then the result shows that  the obtained residuals pretty much follow a normal distribution.

         true

## Midterm Project

(1). Import the weatherHistory.csv into a data frame. How many observations do we have?
```python
import pandas as pd
import numpy as np

data = pd.read_csv('weatherHistory.csv')
data.shape
```
output:
```
(96453, 12)
```

(2). In the weatherHistory.csv data how many features are just nominal variables?
```python
data.head(30)
```
answer:
```
3
```

(3). If we want to use all the unstandardized observations for 'Temperature (C)' and predict the Humidity the resulting root mean squared error is (just copy the first 4 decimal places):
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X = data[['Temperature (C)']].values
y = data[['Humidity']].values

lm = LinearRegression()

# X_train, X_test, y_train, y_test = train_test_split(X, y)

model = lm.fit(X, y)

y_pred = lm.predict(X)

print(np.sqrt(metrics.mean_squared_error(y, y_pred)))
```
output:
```
0.1514437964005473
```

(4). If the input feature is the Temperature and the target is the Humidity and we consider 20-fold cross validations with random_state=2020, the Ridge model with alpha =0.1 and standardize the input train and the input test data. The average RMSE on the test sets is (provide your answer with the first 6 decimal places):
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge as rge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE

X = data['Temperature (C)'].values
y = data['Humidity'].values

df = pd.DataFrame(X, y)

ridge = rge(alpha=0.1)

kf = KFold(n_splits=20, random_state=2020, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
output:
```
0.151438251487059
```

(5). Suppose we want to use Random Forrest with 100 trees and max_depth=50 to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-cross validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 6 decimal places):
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

X = data['Apparent Temperature (C)'].values
y = data['Humidity'].values

df = pd.DataFrame(X, y)

model = RandomForestRegressor(n_estimators=100,max_depth=50)
kf = KFold(n_splits=10, random_state=1693, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
output:
```
0.14350762697638758
```

(6). Suppose we want use polynomial features of degree 6 and we want to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 5 decimal places):
```python
from sklearn.preprocessing import PolynomialFeatures

X = data['Apparent Temperature (C)'].values
y = data['Humidity'].values
  
pf = PolynomialFeatures(degree = 6)
x_poly = pf.fit_transform(X.reshape(-1,1))

df = pd.DataFrame(x_poly, y)

model = LinearRegression()
kf = KFold(n_splits=10, random_state=1693, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
output:
```
0.1434659719585773
```

(7). If the input feature is the Temperature and the target is the Humidity and we consider 10-fold cross validations with random_state=1234, the Ridge model with alpha =0.2. Inside the cross-validation loop standardize the input data. The average RMSE on the test sets is (provide your answer with the first 4 decimal places):
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge as rdg
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE

X = data['Temperature (C)'].values
y = data['Humidity'].values

df = pd.DataFrame(X, y)

ridge = rdg(alpha=0.2)

kf = KFold(n_splits=10, random_state=1234, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
output:
```
0.15144461669728831
```

(8). Suppose we use polynomial features of degree 6 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 4 decimal places):
```python
from sklearn.preprocessing import PolynomialFeatures

X = data.loc[:, ['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']].values
y = data['Temperature (C)'].values
  
pf = PolynomialFeatures(degree = 6)
x_poly = pf.fit_transform(X)

df = pd.DataFrame(x_poly, y)

model = LinearRegression()
kf = KFold(n_splits=10, random_state=1234, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
output:
```
6.119201734148136
```

(9). Suppose we use Random Forest with 100 trees and max_depth=50 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 4 decimal places):
```python
X = data.loc[:, ['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']].values
y = data['Temperature (C)'].values

df = pd.DataFrame(X, y)

model = RandomForestRegressor(n_estimators=100,max_depth=50)
kf = KFold(n_splits=10, random_state=1234, shuffle=True)

i = 0
PE = []

for train_index, test_index in kf.split(df):
    X_train = df.values[train_index]
    y_train = y[train_index]
    X_test = df.values[test_index]
    y_test = y[test_index]
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    PE.append(np.sqrt(MSE(y_test, y_pred)))

print(str(np.mean(PE)))
```
output:
```
5.831136223794951
```      

(10). If we visualize a scatter plot for Temperature (on the horizontal axis) vs Humidity (on the vertical axis) the overall trend seems to be
```python
data.plot.scatter(x = 'Temperature (C)', y = 'Humidity')
```
answer:
```
decreasing
```

## Lab 5

(1). In the case of  kernel Support Vector Machines for classification, such as  the radial basis function kernel,  one or more landmark points are considered by the algorithm.


(2). A hard margin SVM is appropriate for data which is not linearly separable.

(3). In K-nearest neighbors, all observations that fall within a circle with radius of K are included in the estimation for a new point.

(4). For the breast cancer data (from sklearn library), if you choose a test size of 0.25 (25% of your data), with a random_state of 1693, how many observations are in your training set?
```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data
y = data.target

df = pd.DataFrame(data = X, columns = y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1693)

print(len(X_train))
```
output:
```
```

(5). Kernel SVM is only applicable if you have at least 3 independent variables (3 dimensions).

(6). Using your Kernel SVM model with a radial basis function kernel, predict the classification of a tumor if it has a radius mean of 16.78 and a texture mean of 17.89.

(7). Using your logistic model, predict the probability a tumor is malignant if it has a radius mean of 15.78 and a texture mean of 17.89.

(8). Using your nearest neighbor classifier with k=5 and weights='uniform', predict if a tumor is benign or malignant if the Radius Mean  is 17.18, and the Texture Mean is 8.65

(9). Consider a RandomForest classifier with 100 trees, max depth of 5 and random state 1234. From the data consider only the "mean radius" and the "mean texture" as the input features. If you apply a 10-fold stratified cross-validation and estimate the mean AUC (based on the receiver operator characteristics curve) the answer is

(10). What is one reason simple linear regression (OLS) is not well suited to calculating the probability of discrete cases?

(11). When applying the K - Nearest Neighbors classifier we always get better results if the weights are changed from 'uniform' to 'distance'.

## Lab 6

(1). A tree with no depth restriction (an infinite maximum depth) will always attempt to create only pure nodes.
```
true
```

(2). Classification Tree models can be helpful if your data illustrates non-linearity.
```
true
```

(3). In the RandomForestClassifier, what does the n_estimators define?
```
The number of trees to include in the forest.
```

(4). In the lab, based on the test dataset, which model predicted the most false negatives?  Use a max depth of 5 for your trees, and the random state = 1693.  For the random forest classifier, n_estimators should be 1000.
```
Naive Bayes
```

(5). In the lab, what would be considered a "False Positive"?
```
When the model classified something as Malignant, when it was in fact Benign.
```

(6). Naive Bayes classifiers are probabilistic.
```
true
```

(7). The root node of a classification tree is...
```
A node which contains all data points.
```

(8). What is soft classification?
```
A probability-based classification, where an observation is assigned a probability of class membership for each class.
```

(9). What is the Posterior Probability in a Bayesian approach to classification?
```
The probability you are solving for, such as the the probability that the response variable  belongs to a particular class given the data.
```

(10). In a two-class classification the confusion matrix helps with determining
```
The number of true positives, false positives, true negatives and false negatives.
```

(11). The biological function of the "Axon" is represented by what element of a neural network?
```
The outputs from each neuron along the synapse.
```

(12). Match each term to it's correct definition, in the context of a neural network.
```
Neuron - Node in which an activation function is applied
Hidden Layer - A set of neurons that are applied at the same time, going left to right in the network.
Cost function - Measurement of how close predictions are to true values.
Back propagation - Information on accuracy being used to adjust weights or other model parameters.
Gradient Descent - Method for reducing cost function to a minimum.
```

(13). What is an advantage of stochastic gradient descent, as contrasted to traditional gradient descent?
```
It is less likely to get stuck at suboptimal local minima.
```

(14). If we retain only two input features, such as "mean radius" and "mean texture" and apply the Gaussian Naive Bayes model for classification, then the average accuracy determined on a 10-fold cross validation with random_state = 1693 is (do not use the % notation, just copy the first 4 decimals)

output:
```
0.8805
```

(15). From the data retain only two input features, such as "mean radius" and "mean texture" and apply the Random Froest model for classification with 100 trees, max depth of 7 and random_state=1693; The average accuracy determined on a 10-fold cross validation with the same random state is (do not use the % notation, just copy the first 4 decimals)

output:
```
0.8804
```

(16). From the data retain only two input features, such as "mean radius" and "mean texture" we build an Artificial Neural Network for classification that has three hidden layers with 16, 8 and 4 neurons respectively.

Assume that the neurons in the hidden layer have the rectified linear activation ('relu') and the kernel initializer uses the random normal distribution. Assume the output layer has only one neuron with 'sigmoid' activation. 

You will compile the model with 
```
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

and fit the model with 150 epochs and 

validation_split=0.25,batch_size=10,shuffle=False
```
 The average accuracy determined on a 10-fold cross validation (random state=1693) is closer to:

answer:
```
0.89
```

## Lab 7

(1). If we use RandomForest (random_state=310) max_depth=10 and 1000 trees for ranking the importance of the input features the top three features (in decreasing order) are:
```python
import numpy as np
import pandas as pd

from sklearn import datasets

# Declare the columns names 
columns = 'age gender bmi map tc ldl hdl tch ltg glu'.split()  
    
# Call the diabetes dataset from sklearn 
diabetes = datasets.load_diabetes()  
 
# load the dataset as a pandas data frame 
df = pd.DataFrame(diabetes.data, columns=columns)  
      
# define the target variable (dependent variable) as y
y = diabetes.target 

# define independent variable as X
X = pd.DataFrame(diabetes.data, columns=columns)

from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot

model = RandomForestRegressor(random_state=310, max_depth=10, n_estimators = 1000)
model.fit(X, y)

importance = model.feature_importances_

pyplot.bar(columns, importance)
pyplot.show()
```
answer:
```
ltg, bmi, map
```

(2). For the diabetes dataset you worked on the previous question, apply stepwise regression with add/drop p-values both set to 0.001. The model selected has the following input variables:
```python
import statsmodels.api as sm

# Implementation of stepwise regression
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.001, 
                       threshold_out = 0.001, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details """
    
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

answer = stepwise_selection(X,y,[],0.001,0.001)
print(answer)
```
output:
```
Add  bmi                            with p-value 3.46601e-42
Add  ltg                            with p-value 3.03968e-20
Add  map                            with p-value 3.74192e-05
['bmi', 'ltg', 'map']
```

(3). For the diabetes dataset scale the input features by z-scores and then apply the ElasticNet model with alpha=0.1 and l1_ratio=0.5.  If we rank the variables in the decreasing order of the absolute value of the coefficients the top three variables (in order) are:
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score as acc

# scale the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y)

ss = StandardScaler()
Xs_train = ss.fit_transform(X_train)
Xs_test = ss.fit_transform(X_test)

# create and apply model
model = ElasticNet(alpha = 0.1, l1_ratio=0.5)
model.fit(Xs_train, y_train)

y_pred = model.predict(Xs_test)

# ranking
y_pred = abs(y_pred)

print(y_pred)
```
answer:
```
bmi, ltg, map
```

(4). A k-fold cross-validation can be used to determine the best choice of hyper-parameters from a finite set of choices.
answer:
```
true
```

(5). In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data with two features (radius_mean and texture_mean) and, according to what you learned about model selection, you try to determine the best maximum depth (in a range between 1 and 100) and the best  minimum samples per leaf (in a range between 1 and 25) the optimal pair of hyper-parameters (such as max depth and min leaf samples) is:
```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score as acc


tumors = datasets.load_breast_cancer()

X = tumors.data
y = tumors.target
X = X[:,0:2]

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY):
	AC = []
	# prepare cross validation
	kfold = KFold(n_splits=10, random_state=1693, shuffle=True)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = DecisionTreeClassifier(random_state=1693, max_depth=9, min_samples_leaf=22)
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY)
		# evaluate model
		ypred = model.predict(testX)
		# stores scores
		AC.append(acc(testY, ypred))
	return AC

results = evaluate_model(X,y)
print(np.mean(results))
```
ouput:
```
0.8857456140350877
```
answer:
```
(4,23)
```

(6). In this problem consider 10-fold cross-validations and random_state=12345 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data with two features (radius_mean and texture_mean) and, according to what you learned about model selection,  you try to determine the best maximum depth (in a range between 1 and 100) and the best  minimum samples per leaf (in a range between 1 and 25) the number of False Negatives is:
```python
def validation(X,y,k,model):
  accuracy = []
  pipe = Pipeline([('Classifier',model)])
  kf = KFold(n_splits=k,shuffle=True,random_state=1693)
  for idxtrain, idxtest in kf.split(X):
    X_train = X[idxtrain,:]
    y_train = y[idxtrain]
    X_test = X[idxtest,:]
    y_test = y[idxtest]
    pipe.fit(X_train,y_train)
    accuracy.append(accuracy_score(y_test,pipe.predict(X_test)))
  return np.mean(accuracy)
  
model = tree.DecisionTreeClassifier(random_state=12345, max_depth=5, min_samples_leaf=19)
features = ['mean radius', 'mean texture']
X = df[features].values
model.fit(X, y);
pred = model.predict(X)

spc = ['No Default','Default']
cm = confusion_matrix(y, pred)
pd.DataFrame(cm, columns=spc, index=spc)
```
answer:
```
22
```

(7). In this problem consider 10-fold cross-validations and random_state=1693 for cross-validations and the decision tree. If you analyze the data with benign/malign tumors from breast cancer data set with two features (radius_mean and texture_mean) and, according to what you learned about model selection,  you try to determine the best maximum depth (in a range between 1 and 100) and the best  minimum samples per leaf (in a range between 1 and 25) the accuracy is about:
```python
model = tree.DecisionTreeClassifier(random_state=1693, max_depth=5, min_samples_leaf=19)
features = ['mean radius', 'mean texture']
X = df[features].values
print(validation(X, y, 10, model))
```
ouput:
```
0.8892543859649124
```
answer:
```
89%
```
(8). Maximum depth, minimum node size, and learning rate are all examples of what type of parameters?
answer:
```
hyperparameters
```

(9). Selecting an appropriate model requires:
answer:
```
all are correct
```

(10). A good reason for implementing a feature selection technique is:
answer:
```
to create a parsimonious model
```

(11). The concept of Principal Component Analysis refers to:
answer:
```
Determining the directions along which we maximize the variance of the input features.
```

(12). In this problem the input features will be scaled by the z-scores and consider a use a random_state=1234. If you analyze the data with benign/malign tumors from breast cancer data, consider a decision tree with max_depth=10,min_samples_leaf=20 and fit on 9 principal components the number of true positives is:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

tumors = datasets.load_breast_cancer()

X = tumors.data
y = tumors.target

dt = DecisionTreeClassifier(random_state=1234, max_depth=10, min_samples_leaf=20)
pca = PCA(n_components=9)

ss = StandardScaler()
Xs = ss.fit_transform(X)
 
pca.fit_transform(Xs)
```
answer:
```
196
```
