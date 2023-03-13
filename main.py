from Classes import *

def test():
    a = 0.5
    k = 1
    sigma = 0.25
    T = 1
    n = 20
    x0 = 1
    scheme_implicit_3 = Implicit_3(x0, a, k, sigma, T, n)
    print(scheme_implicit_3)
    scheme_implicit_3.simulate()
    print(scheme_implicit_3.simulation)


test()
