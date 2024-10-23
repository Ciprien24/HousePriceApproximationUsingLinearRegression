import numpy as np
import matplotlib.pyplot as plt
from utils import *
from public_tests import *
import copy
import math

# Load the Data (you'll need to define this function to load x_train and y_train)
x_train, y_train = load_data()

# Convert x_train and y_train to numpy arrays (in case they're not already)
x_train = np.array(x_train)
y_train = np.array(y_train)

# View the data
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Type of x_train: {type(x_train)}")
print(f"Type of y_train: {type(y_train)}")

m = len(x_train)
print(f"Number of training examples: {m}")

# Plot the data
plt.scatter(x_train, y_train, marker='x', color='red')
plt.title("Profits vs Population per city")
plt.ylabel("Profit in $10,000")
plt.xlabel("Population in 10,000s")
plt.show()


# Compute the cost function
def compute_cost(w, b, x, y):
    m = x.shape[0]  # number of training examples
    total_cost = 0
    for i in range(m):
        f_wb = (x[i] * w) + b  # Prediction
        total_cost += (f_wb - y[i]) ** 2  # Squared error
    total_cost = total_cost / (2 * m)  # Mean squared error
    return total_cost


# Compute gradient
def compute_gradient(w, b, x, y):
    m = x.shape[0]  # number of training examples
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = (x[i] * w) + b  # Prediction
        error = f_wb - y[i]  # Error in prediction
        dj_dw += error * x[i]  # Gradient for w
        dj_db += error  # Gradient for b
    dj_dw = dj_dw / m  # Average gradient for w
    dj_db = dj_db / m  # Average gradient for b
    return dj_dw, dj_db


# Gradient Descent implementation
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    m = len(x)
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        # Calculate the gradient
        dj_dw, dj_db = gradient_function(w, b, x, y)

        # Update the parameters using the gradients
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Compute and record the cost every iteration
        if i < 100000:
            cost = cost_function(w, b, x, y)
            J_history.append(cost)

        # Print the cost at intervals (e.g., every 10% of the iterations)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history


# Initialize fitting parameters
initial_w = 0.  # initial weight
initial_b = 0.  # initial bias

# Gradient descent settings
iterations = 1500
alpha = 0.01

# Run gradient descent
w, b, _ = gradient_descent(x_train, y_train, initial_w, initial_b,
                           compute_cost, compute_gradient, alpha, iterations)

print(f"w, b found by gradient descent: w = {w}, b = {b}")

m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b
# Plot the linear fit
plt.plot(x_train, predicted, c = 'b')
plt.title("Profits vs Population per city")
plt.ylabel("Profit in $10,000")
plt.xlabel("Population in 10,000s")
plt.scatter(x_train, y_train, marker='x', color='red')
plt.show()

# Predicting for special values
predict1 = 3.5 * w + b
print(predict1 * 10000)
predict2 = 7.5 * w + b
print(predict2 * 10000)


