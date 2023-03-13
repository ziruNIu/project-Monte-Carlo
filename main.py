from Classes import *
import matplotlib.pyplot as plt

def test():
    a = 1
    k = 1
    sigma = np.sqrt(3)
    T = 1
    n = 200
    x0 = 0.5
    lambdaa = sigma**2 / 8
    scheme_implicit_3 = Implicit_3(x0, a, k, sigma, T, n)
    scheme_implicit_4 = Implicit_4(x0, a, k, sigma, T, n)
    scheme_E_lambda = E_lambda(x0, a, k, sigma, lambdaa, T, n)
    scheme_E_0 = E_lambda(x0, a, k, sigma, 0, T, n)
    scheme_implicit_4.simulate()
    scheme_implicit_3.simulate()
    scheme_E_lambda.simulate()
    scheme_E_0.simulate()
    time_grid = [i*T/n for i in range(n+1)]
    plt.plot(time_grid, scheme_implicit_3.simulation, color = 'b')
    plt.plot(time_grid, scheme_implicit_4.simulation, color = 'r')
    plt.plot(time_grid, scheme_E_lambda.simulation, color = 'g')
    plt.plot(time_grid, scheme_E_0.simulation, color = 'y')
    plt.legend(["Implicit 3", "Implicit 4", "E(σ2/8)", "E(0)"])
    plt.show()




test()
