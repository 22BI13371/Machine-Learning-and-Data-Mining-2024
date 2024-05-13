import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

file = "Practice2/marks.txt"
header = []

with open(file, 'r') as f:
    data = pd.read_csv(f, header=None, names=[
                       'exam1', 'exam2', 'result', 'x0'])


data['x0'] = 1
# print(data.head())
X = data[['x0', 'exam1', 'exam2']]
Y = data['result']
# Y = np.array((Y-Y.mean())/Y.std())
# X = X.apply(lambda rec: (rec - rec.mean())/rec.std(), axis=0)

# print(Y)


def initialize_theta(dim):
    theta = np.random.rand(dim)
    return theta


# theta = initialize_theta(3)


def predict_Y(theta, X):
    return 1/(1 + np.power(np.e, -np.dot(X, theta)))


# Y_hat = predict_Y(theta, X)

# print(X.shape, Y.shape, theta.shape)

# Y_hat is model prediction


def get_cost(Y, Y_hat):
    return np.sum(np.dot(Y.T, np.log10(Y_hat)) + np.dot((1 - Y).T, np.log10(1 - Y_hat))/-len(Y))


def update_theta(y_hat, theta_0, learning_rate):
    dw = np.dot(y_hat.T, (1 - y_hat))
    theta_1 = theta_0 - learning_rate * dw
    return theta_1


def gradien_descent(X, Y, alpha, num_iterations):
    theta = initialize_theta(X.shape[1])
    print(theta)
    iter_num = 0
    gd_iterations_data = pd.DataFrame(columns=['iteration',  'cost'])
    result_idx = 0
    for each_iter in range(num_iterations):
        Y_hat = predict_Y(theta, X)
        this_cost = get_cost(Y, Y_hat)
        prev_theta = theta
        theta = update_theta(Y_hat, prev_theta, alpha)
        if (iter_num % 10 == 0):
            gd_iterations_data.loc[result_idx] = [iter_num, this_cost]
            result_idx += result_idx + 1
        iter_num += 1
    print("theta at the end: ", theta)
    return gd_iterations_data, theta


gd_iteration_data, theta = gradien_descent(
    X, Y, alpha=0.01, num_iterations=200)

plt.plot(gd_iteration_data['iteration'], gd_iteration_data['cost'])
plt.xlabel("number of iterations")
plt.ylabel("cost or MSE")
plt.show()
print(gd_iteration_data.tail())
