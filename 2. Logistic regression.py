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

# Initialising Variables
num_of_iterations = 500 #number of iterations
m = 400 #number of training examples
cost_history = [] #cost function
w = np.zeros((1, 2)) #weights
b = 0 #bias
dw = np.zeros((1, 2)) # gradient of weights
db = 0 # gradient of bias

# Gradient Descent
for i in range(num_of_iterations):
    # Forward Pass
    z = np.dot(w, x_scaled.T).T + b
    y_pred= 1 / (1 + np.exp(-z))
    
    # Cost Function
    cost = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    cost_history.append(cost)
    
    # Backward Pass
    dw = dw + (1 / m) * np.dot((y_pred - y).T, x_scaled)
    db = db + np.sum(y_pred - y) / m
    
    # Update weights and bias
    w = w - 0.0001 * dw
    b = b - 0.0001 * db
    
    if i % 100 == 0:
        print(f"Iteration {i}, Cost Function: {cost}")
    
# Plot Cost Function
plt.plot(cost_history)
plt.title("Cost Function")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

# Predictions vs Actuals
z = np.dot(w, x_scaled.T).T + b
y_pred= 1 / (1 + np.exp(-z))
for i in range(400):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
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
