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
