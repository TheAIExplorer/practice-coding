import numpy as np
y = 1
y_pred = 0.9
cost = -1 * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
print(cost)
