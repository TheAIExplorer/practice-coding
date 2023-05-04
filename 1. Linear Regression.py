# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# # Read CSV file
# df = pd.read_csv('50_Startups.csv')

# # Select first 3 column as features and last column as target variable
# x = df.iloc[:, :3].values
# y = df.iloc[:, -1].values.reshape(-1, 1)

# Generate random input features with 3 columns and 100 rows
x = np.random.rand(100, 3)
# Generate random weights for the linear function
w = np.random.rand(3, 1)
# Generate random bias for the linear function
b = np.random.rand(1)
# Calculate the linear target variable
y = np.dot(x, w) + b

# Feature scaling
x_scaled = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Initialising Variables
num_of_iterations = 1100 #number of iterations
m = 50 #number of training examples
cost_history = [] #cost function
w = np.zeros((1, 3)) #weights
b = 0 #bias
dw = np.zeros((1, 3)) # gradient of weights
db = 0 # gradient of bias
lmbd= 0.01 # regularization term

# Gradient Descent
for i in range(num_of_iterations):
    # Forward Pass
    y_pred = np.dot(w, x_scaled.T).T + b
    
    # Cost Function with regularization
    cost = np.sum((1 / (2 * m)) * ((y_pred - y)**2) + lmbd*(w**2))
    cost_history.append(cost)
    
    # Backward Pass
    dw = dw + (1 / m) * (np.dot((y_pred - y).T, x_scaled) + lmbd*w)
    db = db + np.sum(y_pred - y) / m
    
    # Update weights and bias
    w = w - 0.000001 * dw
    b = b - 0.000001 * db
    
    if i % 100 == 0:
        print(f"Iteration {i}, Cost Function: {cost}")
    
# Plot Cost Function
plt.plot(cost_history)
plt.title("Cost Function")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# Predictions vs Actuals
y_pred = np.dot(w, x_scaled.T).T + b
plt.scatter(range(len(y)), y, color="blue")
plt.scatter(range(len(y_pred)), y_pred, color="red")
plt.legend(["Actuals", "Predictions"])
plt.title("Predictions vs Actuals")
plt.xlabel("Index")
plt.ylabel("Profit")
plt.show()

# 3D Surface Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x1 = x_scaled[:, 0]
x2 = x_scaled[:, 1]
y_pred = y_pred.reshape((100,))
ax.plot_trisurf(x1, x2, y_pred, cmap='viridis')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
plt.show()

# # Take input from user
# x1, x2, x3 = input("Input the values of features x1, x2, x3: ").split()

# # Convert input to float
# x_to_predict = [float(x1), float(x2), float(x3)]

# # Make prediction
# y_pred = np.dot(w, x_to_predict) + b

# print('the prediction is:', y_pred)
# %%
