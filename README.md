## Lab 1

## Lab 2

## Lab 3

1. An "ordinary least squares" (or OLS) model seeks to minimize the differences between your true and estimated dependent variable.

         True



2. Do you agree or disagree with the following statement: In a linear regression model, all feature must correlate with the noise in order to obtain a good fit.

         Disagree, the noise refers to features that do no correlate well, so less noise means there is a better fit.

   

3. Write your own code to import L3Data.csv into python as a data frame. Then save the feature values 'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234. If we use the features of x to build a multiple linear regression model for predicting y then the root mean square error on the test data is close to:

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


4. In practice we determine the weights for linear regression with the "X_test" data.

         False.


5. Polynomial regression is best suited for functional relationships that are non-linear in weights.

         False


6. Linear regression, multiple linear regression, and polynomial regression can be all fit using LinearRegression() from the sklearn.linear_model module in Python.

         True



7. Write your own code to import L3Data.csv into python as a data frame. Then save the feature values 'days online','views','contributions','answers' into a matrix x and consider 'Grade' values as the dependent variable. If you separate the data into Train & Test with test_size=0.25 and random_state = 1234, then the number of observations we have in the Train data is

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

8. The gradient descent method does not need any hyperparameters.

         False


9. To create and display a figure using matplotlib.pyplot that has visual elements (scatterplot, labeling of the axes, display of grid), in what order would the below code need to be executed?

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


10. Which of the following forms is *not linear in the weights* ?

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

                  The option with "e" and exponents is not linear.
