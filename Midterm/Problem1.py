import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fmin


def t(x):
    return np.exp(-x * x) - np.exp(-(x - 5) * (x - 5))


def kernel(x1, x2):
    return np.exp(-(x1 - x2) ** 2 / 2)


def mean(x, CT_n, X_n):
    result = 0
    for i in range(0, len(X_n)):
        result = result + kernel(x, X_n[i]) * CT_n[i]
    return -result


def variance(x, C_n, X_n):
    k = np.ones(len(X_n))
    for i in range(0, len(X_n)):
        k[i] = kernel(x, X_n[i])
    return 1 - np.dot(np.dot(k.transpose(), np.linalg.pinv(C_n)), k)


def utility(x, C_n, T_n, X_n):
    C_ninv = np.linalg.inv(C_n)
    CT_n = np.dot(C_ninv, T_n)
    # compute mean
    mu = 0
    for i in range(0, len(X_n)):
        mu = mu + kernel(x, X_n[i]) * CT_n[i]
    # compute variance
    k = np.ones(len(X_n))
    for i in range(0, len(X_n)):
        k[i] = kernel(x, X_n[i])
    var = 1 - np.dot(np.dot(k.transpose(), C_ninv), k)
    return -mu - var


x1 = -0.8
x2 = 5.7
max = 12

# covariance matrix
C = np.ones((max, max))
# f(x)
T = np.ones(max)
# x_k
X = np.ones(max)

# initial evaluation
X[0] = x1
X[1] = x2
T[0] = t(x1)
T[1] = t(x2)
C[0, 1] = kernel(x1, x2)
C[1, 0] = C[0, 1]

fig = plt.figure(figsize=(15, 20))

# Bayesian optimization
for i in range(2, max):
    # slice current covariance matrix and data
    C_n = C[:i, :i]
    T_n = T[:i]
    X_n = X[:i]
    CT_n = np.dot(np.linalg.pinv(C_n), T_n)
    # find next best point to evaluate
    xn = fmin(utility, 3, args=(C_n, T_n, X_n), disp=False)[0]
    # augment data set and covariance matrix
    X[i] = xn
    T[i] = t(xn)
    for j in range(0, i + 1):
        C[i, j] = kernel(X[j], xn)
        C[j, i] = C[i, j]
    # plot current state
    xp = np.arange(-3, 8, 0.1)
    yp = -mean(xp, CT_n, X_n)
    varp = np.zeros(len(xp))
    ax = fig.add_subplot(5, 2, (i - 1))
    ax.set_ylim((-1.5, 1.5))
    for j in range(0, len(xp)):
        varp[j] = variance(xp[j], C_n, X_n)
        varp[j] = math.sqrt(math.fabs(varp[j]))
    ax.plot(xp, yp, color='b')
    ax.fill_between(xp, yp + varp, yp - varp, color='y', alpha=0.2)

    for j in range(0, i):
        ax.plot(X[j], T[j], color='g', marker='o')

    ax.plot(X[i], T[i], color='r', marker='v')
    ax.set_title("plot at " + str(i - 1) + "th iteration")
plt.show()

print(X)
