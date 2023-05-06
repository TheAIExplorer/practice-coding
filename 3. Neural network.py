# %%
# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Reading the CSV File
df = pd.read_csv('Social_Network_Ads.csv')

# Select first 3 columns as features and last column as target variable
x = df.iloc[:, :2].values
y = df.iloc[:, -1].values.reshape(-1, 1)

# Feature scaling
x_scaled = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Initializing Variables
num_of_iterations = 2500 #number of iterations
m = 400 #number of training examples
cost_history = [] #cost function

# Initializing weights and biases
w1 = np.zeros(2, 3) # weights for first hidden layer
b1 = np.zeros((1, 3)) # biases for first hidden layer
w2 = np.zeros(3, 1) # weights for output layer
b2 = 0 # bias for output layer

# Gradient Descent
for i in range(num_of_iterations):
    # Forward Pass
    z1 = np.dot(x_scaled, w1) + b1
    a1 = 1 / (1 + np.exp(-z1)) # activation function for first hidden layer
    z2 = np.dot(a1, w2) + b2
    y_pred = 1 / (1 + np.exp(-z2))  # activation function for Second hidden layer
    
    # Cost Function
    cost = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    cost_history.append(cost)
    
    # Backward Pass
    dz2 = y_pred - y
    dw2 = 1/m * np.dot(a1.T, dz2)
    db2 = 1/m * np.sum(dz2)
    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * a1 * (1 - a1) # derivative of sigmoid activation function
    dw1 = 1/m * np.dot(x_scaled.T, dz1)
    db1 = 1/m * np.sum(dz1, axis=0)
    
    # Update weights and biases
    w2 = w2 - 0.1 * dw2
    b2 = b2 - 0.1 * db2
    w1 = w1 - 0.1 * dw1
    b1 = b1 - 0.1 * db1
    
    if i % 100 == 0:
        print(f"Iteration {i}, Cost Function: {cost}")
    
# Plot Cost Function
plt.plot(cost_history)
plt.title("Cost Function")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# Predictions vs Actuals
z1 = np.dot(x_scaled, w1) + b1
a1 = 1 / (1 + np.exp(-z1))
z2 = np.dot(a1, w2) + b2
y_pred = 1 / (1 + np.exp(-z2))
for i in range(400):
    if y_pred[i] >= 0.5:
        y_pred[i] = 2
    else:
        y_pred[i] = 0

# Predictions Accuracy percentage
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")

# Plotting the Graphs
plt.scatter(range(len(y)), y, color="blue")
plt.legend(["Actuals"])
plt.title("Actuals")
plt.xlabel("Index")
plt.ylabel("Profit")
plt.show()
plt.scatter(range(len(y_pred)), y_pred, color="red")
plt.legend(["Predictions"])
plt.title("Predictions")
plt.xlabel("Index")
plt.ylabel("Profit")
plt.show()
# %%

%%
