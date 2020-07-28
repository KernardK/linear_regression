from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from matplotlib import style
from statistics import mean


import matplotlib.pyplot as plt
import numpy as np


# bestFitIntercepts
def bestFitIntercepts(x, y):
    m = (((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) * mean(x)) - mean(x * x)))
    b = mean(y) - m * mean(x)
    return m, b


# Load data form diabetes
diabetes = load_diabetes()

# X and Y data
X = diabetes.data[:, np.newaxis, 2]
Y = diabetes.target


# Splits the data into
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Reshape the x and y array to remove an extra dimension of 1 from its shape
x = np.squeeze(X_train)
y = np.squeeze(Y_train)

# Calls the function for the slope and the intercepts
m, b = bestFitIntercepts(x, y)

# Recalls the formula to calculate the regression line
regression_line = [(m * x) + b for x in X_train]

# Predicted value for X
predict_X = 7

# Predict the value for Y
predict_Y = (m * predict_X) + b


# Styles the graph
style.use("ggplot")

# Plots the graph, with the labels and color
plt.scatter(X_train, Y_train, color="red", label='Training Data')
plt.scatter(X_test, Y_test, color="green", label='Test Data')
plt.plot(X_train, regression_line, color="blue", label='Regression Line')

# Creates the legend
plt.legend()

# Shows the graph
plt.show()