from scheme import *
import pandas as pd

def strong_error(scheme, scheme_type, steps, M = 10000):
    result = []
    for N in steps:
        dW_2 = np.sqrt(scheme.T / N) * rng.standard_normal((N, M))
        dW_1 = np.array([dW_2[2*i]+dW_2[2*i+1] for i in range(N//2)]) 
        scheme_paths_1= scheme.paths_euler(scheme_type, dW_1)
        scheme_paths_2 = scheme.paths_euler(scheme_type, dW_2)
        error = scheme_paths_1 - np.array([scheme_paths_2[2*i] for i in range(N//2 + 1)])
        E = np.max(error**2, axis = 0)
        I_M = np.mean(E)
        size_IC_M = 1.96 * np.std(E) / np.sqrt(M)
        result.append([np.sqrt(I_M), np.sqrt(I_M-size_IC_M), np.sqrt(I_M+size_IC_M)])
    return np.array(result)



def weak_error(f, scheme, scheme_type, steps, M = 10000):
    """
    here the function must allow the vectorized computation
    """
    result = []
    for N in steps:
        dW_2 = np.sqrt(scheme.T / N) * rng.standard_normal((N, M))
        dW_1 = np.array([dW_2[2*i]+dW_2[2*i+1] for i in range(N//2)]) 
        paths_1 = scheme.paths_euler(scheme_type, dW_1)
        paths_2 = scheme.paths_euler(scheme_type, dW_2)
        sample_1 = f(paths_1[-1])
        sample_2 = f(paths_2[-1])
        mean_1 = sample_1.mean()
        mean_2 = sample_2.mean()
        result.append([mean_1, mean_2])

    return np.array(result)
        


