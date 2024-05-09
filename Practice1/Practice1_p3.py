import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

file = "Practice1/Advertising.csv"
header = []

with open(file, 'r') as f:
    data = pd.read_csv(f)
data['x0'] = 1
# print(data.head())
X = data[['x0', 'TV', 'radio', 'newspaper']]
Y = data['sales']
Y = np.array((Y-Y.mean())/Y.std())
X = X.apply(lambda rec: (rec - rec.mean())/rec.std(), axis=0)
X['x0'] = 1

# print(X)


def initialize_theta(dim):
    theta = np.random.rand(dim)
    return theta


# theta = initialize_theta(4)


def predict_Y(theta, X):
    return np.dot(X, theta)


# Y_hat = predict_Y(theta, X)

# print(X.shape, Y.shape, theta.shape)


def get_cost(Y, Y_hat):
    Y_resd = Y - Y_hat
    return np.sum(np.dot(Y_resd.T, Y_resd))/(2 * len(Y-Y_resd))


def update_theta(x, y, y_hat, theta_0, learning_rate):
    dw = (np.dot((y_hat - y), x)*2)/len(y)
    theta_1 = theta_0 - learning_rate * dw
    return theta_1


# print(theta)
# print(get_cost(Y, Y_hat))
# print(update_theta(X, Y, Y_hat, theta, 0.01))


# def gradien_descent(X, Y, alpha, num_iterations):
#     theta = initialize_theta(X.shape[1])
#     iter_num = 0
#     gd_iterations_data = pd.DataFrame(columns=['iteration',  'cost'])
#     result_idx = 0
#     for each_iter in range(num_iterations):
#         Y_hat = predict_Y(theta, X)
#         this_cost = get_cost(Y, Y_hat)
#         prev_theta = theta
#         theta = update_theta(X, Y, Y_hat, prev_theta, alpha)
#         if (iter_num % 10 == 0):
#             gd_iterations_data.loc[result_idx] = [iter_num, this_cost]
#             result_idx += result_idx + 1
#         iter_num += 1
#     print("theta at the end: ", theta)
#     return gd_iterations_data, theta


# gd_iteration_data, theta = gradien_descent(
#     X, Y, alpha=0.01, num_iterations=200)

# plt.plot(gd_iteration_data['iteration'], gd_iteration_data['cost'])

def normal_equation(X, Y):
    theta = initialize_theta(4)
    prev_theta = theta
    X_transpose = np.transpose(X)
    temp2 = np.dot(X_transpose, X)
    temp1 = np.linalg.inv(temp2)

    # theta = np.dot(np.dot(np.linalg.inv(
    #     np.dot(X_transpose, X)), X_transpose), Y)
    return theta


Y = Y
X = X.to_numpy
theta = normal_equation(X, Y)
# plt.xlabel("number of iterations")
# plt.ylabel("cost or MSE")
# plt.show()
# print(gd_iteration_data.tail())
