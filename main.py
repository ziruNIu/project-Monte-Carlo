from scheme import *
import matplotlib.pyplot as plt
from errors import *

def test():
    a = 1
    k = 1
    sigma = 1
    T = 1
    x0 = 1
    lambdaa = sigma**2 / 8
    scheme = CIR(x0, a, k, sigma, T, lambdaa)
    steps = [100*i for i in range(1,20)]
    results_3 = strong_error(scheme, "implicit_3", steps)
    results_4 = strong_error(scheme, "implicit_4", steps)
    results_lambda = strong_error(scheme, "E_lambda", steps)
    results_0 = strong_error(scheme, "E_0", steps)
    n_grid = [1/x for x in steps][::-1]
    plt.plot(n_grid, list(results_3[:,0])[::-1], color='b')
    plt.plot(n_grid, list(results_4[:,0])[::-1], color='g')
    plt.plot(n_grid, list(results_lambda[:,0])[::-1], color='r')
    plt.plot(n_grid, list(results_0[:,0])[::-1], color='y')
    plt.legend(["Implicit 3", "Implicit 4", "E(σ2/8)", "E(0)"])
    plt.show()




test()
