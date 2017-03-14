#problem2-10

import numpy as np
from scipy.misc import derivative
import matplotlib.pyplot as plt


def Phi(x, m):
    """
    return m-th order polynormial basis design matrix of x
    """
    phi = np.ones((len(x), m + 1))
    for i in range(0, len(x)):
        for j in range(0, m + 1):
            phi[i, j] = x[i] ** j
    return phi


def Phid(x, m):
    """
    return derivative of m-th order polynormial basis design matrix of x
    """
    phid = np.zeros((len(x), m + 1))
    for i in range(0, len(x)):
        for j in range(1, m + 1):
            phid[i, j] = j * x[i] ** (j - 1)
    return phid


def ML_f(phi, t, phi_new):
    """
    return maximum likelihood prediction of f(x_new).
    :param phi: design matrix
    :param t: observation
    :param phi_new: new data for prediction
    """
    pp_inv = np.linalg.inv(np.dot(phi.transpose(), phi))
    py = np.dot(phi.transpose(), t)
    w = np.dot(pp_inv, py)
    f = np.dot(phi_new, w)
    return f


def Bayesian_f(phi, t, phi_new, a, b):
    """
    return bayesian prediction of f(x_new) and the variance of prediction
    :param phi: design matrix
    :param t: observation
    :param phi_new: new data for prediction
    :param a: variance of observation
    :param b: variance of bayesian prior
    :return:
    """

    var = []
    S_inv = 1/b * np.identity(phi.shape[1]) + 1/a * np.dot(phi.transpose(), phi)
    S = np.linalg.inv(S_inv)
    m = 1/a * np.dot(np.dot(S, phi.transpose()), t)
    f = np.dot(phi_new, m)
    for i in range(0, len(phi_new)):
        variance = a + np.dot(np.dot(phi_new[i,:].transpose(),S), phi_new[i, :].transpose())
        var.append(variance)
    return f, var




a = 0.1  # noise variance
b = 1
m = 4  # order of polynormial bases
# generate 11 data points D(x,y) of equally spaced samples x[0, 3], t are noisy observations of sinc function
x = np.arange(0, 3.1, 0.3)
t = np.sinc(x) + np.random.normal(0, a, len(x))

# generate 5 derivative data points Dd(xd,yd) equally spaced samples x[0, 3], t are noisy observations of derivative sinc function
xd = np.arange(0.1, 3.1, 0.72)
td = derivative(np.sinc, xd) + np.random.normal(0, a, len(xd))

# generate 100 equally spaced points between x[0,3]
x_new = np.arange(0, 3.0, 0.03)

phi = Phi(x, m)
phid = Phid(xd, m)
phid_new = Phid(x_new, m)
phi_new = Phi(x_new, m)


# maximum likelihood prediction of f' using D(x,y)
fd1_ML = ML_f(phi, t, phid_new)
td1_ML = ML_f(phi, t, phid)
std1_ML = np.sqrt(np.average(np.absolute(td1_ML - td)))

# maximum likelihood prediction of f' using D(x,y) and Dd(xd, yd)
fd2_ML = ML_f(np.concatenate((phi, phid), axis=0), np.concatenate((t, td)), phid_new)
td2_ML = ML_f(np.concatenate((phi, phid), axis=0), np.concatenate((t, td)), phid)
std2_ML = np.sqrt(np.average(np.absolute(td2_ML - td)))

# bayesian prediction of f' using D(x,y)
fd1_Bay, var1 = Bayesian_f(phi, t, phid_new, a, b)
std1_Bay = np.sqrt(np.absolute(var1))

# bayesian prediction of f' using D(x,y) and Dd(xd, yd)
fd2_Bay, var2 = Bayesian_f(np.concatenate((phi, phid)), np.concatenate((t, td)), phid_new, a, b)
std2_Bay = np.sqrt(np.absolute(var2))

fig = plt.figure()

ax = fig.add_subplot(221)
ax.set_xlim((0,3))
ax.set_ylim((-2,1.5))
ax.set_title('ML prediction and uncertainty of\nderivative of sinc function using D')
ax.plot(x_new, derivative(np.sinc, x_new), color='b')
ax.fill_between(x_new, fd1_ML - std1_ML, fd1_ML + std1_ML, color='g', alpha=0.2)
ax.plot(x_new, fd1_ML, color='g')
ax.scatter(xd, td, marker='o')

ax = fig.add_subplot(222)
ax.set_xlim((0,3))
ax.set_ylim((-2,1.5))
ax.set_title('ML prediction and uncertainty of\nderivative of sinc function using D and D_d')
ax.plot(x_new, derivative(np.sinc, x_new), color='b')
ax.fill_between(x_new, fd2_ML - std2_ML, fd2_ML + std2_ML, color='g', alpha=0.2)
ax.plot(x_new, fd2_ML, color='g')
ax.scatter(xd, td, marker='o')

ax = fig.add_subplot(223)
ax.set_xlim((0,3))
ax.set_ylim((-2,1.5))
ax.set_title('ML prediction and uncertainty of\nderivative of sinc function using D')
ax.plot(x_new, derivative(np.sinc, x_new), color='b')
ax.fill_between(x_new, fd1_Bay - std1_Bay, fd2_Bay + std1_Bay, color='r', alpha=0.2)
ax.plot(x_new, fd1_Bay, color='r')
ax.scatter(xd, td, marker='o')

ax = fig.add_subplot(224)
ax.set_xlim((0,3))
ax.set_ylim((-2,1.5))
ax.set_title('Bayesian prediction and uncertainty of\nderivative of sinc function using D and D_d')
ax.plot(x_new, derivative(np.sinc, x_new), color='b')
ax.fill_between(x_new, fd2_Bay - std2_Bay, fd2_Bay + std2_Bay, color='r', alpha=0.2)
ax.plot(x_new, fd2_Bay, color='r')
ax.scatter(xd, td, marker='o')

plt.show()


