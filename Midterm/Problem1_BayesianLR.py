import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fmin


def t(x, alpha):
    return np.exp(-x * x) - np.exp(-(x - 5) * (x - 5)) + np.random.normal(0, alpha)


def base(x, mu, sigma):
    return 1/np.sqrt(2*sigma*np.pi)*np.exp(-(x-mu)**2/(2*sigma))


def mean(Phi, m_n):
    return np.dot(Phi, m_n)


def variance(Phi, S_n, alpha):
    var = np.zeros(len(Phi))
    for i in range(0, len(Phi)):
        var[i] = np.dot(np.dot(Phi[i].transpose(), S_n), Phi[i]) + alpha
    return var


def utility(x, X, m_n, S_n, sigma):
    phi = np.zeros(len(X))
    for i in range(0, len(X)):
        phi[i] = base(x, X[i], sigma)
    return - np.dot(m_n.transpose(), phi) - np.dot(np.dot(phi.transpose(), S_n), phi)


x1 = -0.8
x2 = 5.7
max = 4
sigma = 1
beta = 10
alpha = 0.01
# design matrix
Phi = np.ones((max, max))
# f(x)
T = np.ones(max)
# x_k
X = np.ones(max)

# initial evaluation
X[0] = x1
X[1] = x2
T[0] = t(x1, alpha)
T[1] = t(x2, alpha)
for i in range(0,2):
    for j in range(0, 2):
        Phi[i, j] = base(X[i], X[j], sigma)

fig = plt.figure(figsize=(15, 20))

# Bayesian optimization
for i in range(2, max):
    # slice current covariance matrix and data
    Phi_n = Phi[:i, :i]
    T_n = T[:i]
    X_n = X[:i]
    S_n_inv = 1/beta*np.identity(i) + 1/alpha*np.dot(Phi_n.transpose(), Phi_n)
    S_n = np.linalg.inv(S_n_inv)

    m_n = 1/alpha*np.dot(np.dot(S_n, Phi_n.transpose()), T_n)
    # find next best point to evaluate
    xn = fmin(utility, 3, args=(X_n, m_n, S_n, sigma), disp=False)[0]
    # augment data set and design matrix
    X[i] = xn
    T[i] = t(xn, alpha)
    for j in range(0, i + 1):
        Phi[i, j] = base(xn, X[j], beta)
        Phi[j, i] = base(X[j], xn, beta)
    # plot current state
    xp = np.arange(-3, 8, 0.1)
    Phi_xp = np.zeros((len(xp), i))
    for j in range(0, len(xp)):
        for k in range(0, i):
            Phi_xp[j, k] = base(xp[j], X[k], sigma)
    print(Phi_xp)
    yp = mean(Phi_xp, m_n)

    varp = variance(Phi_xp, S_n, alpha)
    ax = fig.add_subplot(5, 2, (i - 1))
    ax.set_ylim((-1.5, 1.5))
    ax.plot(xp, yp, color='b')
    ax.fill_between(xp, yp + varp, yp - varp, color='y', alpha=0.2)

    for j in range(0, i):
        ax.plot(X[j], T[j], color='g', marker='o')

    ax.plot(X[i], T[i], color='r', marker='v')
    ax.set_title("plot at " + str(i - 1) + "th iteration")
plt.show()


