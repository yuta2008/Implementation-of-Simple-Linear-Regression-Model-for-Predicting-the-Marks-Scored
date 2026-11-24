# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn.
2. Load the dataset containing the number of study hours and marks scored by students.
3. Split the dataset into training and testing sets, train the Simple Linear Regression model, and predict the marks for the test data.
4. Display the regression equation, evaluate the model, and plot the regression line with the data points.

 

## Program:
```
  import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Sample Dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [18, 25, 35, 45, 52, 60, 68, 78, 85, 95]
}

df = pd.DataFrame(data)

# Independent and Dependent variables
X = df[['Hours']]

y = df['Marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()

model.fit(X_train, y_train)

# Predict marks
y_pred = model.predict(X_test)

# Regression equation
print("Slope (m):", model.coef_[0])

print("Intercept (b):", model.intercept_)

print(f"Regression Equation: Marks = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")

# Predict marks for a new student
hours = 7.5

predicted_marks = model.predict([[hours]])

print(f"\nPredicted Marks for {hours} study hours = {predicted_marks[0]:.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')

plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.xlabel("Study Hours")

plt.ylabel("Marks")

plt.title("Simple Linear Regression - Marks Prediction")

plt.legend()

plt.grid(True)

plt.show()
```

## Output:
<img width="1349" height="561" alt="image" src="https://github.com/user-attachments/assets/2c78a41d-04f1-4da0-879e-6568580314c3" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
