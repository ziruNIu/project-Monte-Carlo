from scheme import *
import pandas as pd

def strong_error(scheme, scheme_type, steps, M = 10000):
    """ Computes the strong error for the chosen scheme type

        Args : 
            scheme :  instance of CIR class with chosen parameters
            scheme_type (string) : from the 5 types of CIR schemes 
            steps (list<int>) : list of discretization steps
            M (int) : number of simulation trajectories

        Returns :
            numpy.array : array of the strong errors of the chosen 
                          scheme_type for every step
        """
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


def speed_convergence(scheme, scheme_type, N = 200, M = 10000):
    """ Computes the speed of convergence for the chosen scheme type

        Args : 
            scheme :  instance of CIR class with chosen parameters
            scheme_type (string) : from the 5 types of CIR schemes 
            N (int) : discretization steps
            M (int) : number of simulation trajectories

        Returns :
            numpy.array : array of the speeds of convergence of the chosen 
                          scheme_type for every sigma^2/2a
        """
    result = []
    sigma_grid = [0.1*i for i in range(25)]
    for sigma in sigma_grid:
        scheme.sigma = sigma
        dW_2_10n = np.sqrt(scheme.T / (20*N)) * rng.standard_normal((20*N, M))
        dW_1_10n = np.array([dW_2_10n[2*i]+dW_2_10n[2*i+1] for i in range(10*N)])
        dW_2_n = np.array([dW_2_10n[5*i]+dW_2_10n[5*i+1] + dW_2_10n[5*i+2]+dW_2_10n[5*i+3] + dW_2_10n[5*i+4] for i in range(2*N)])
        dW_1_n = np.array([dW_2_n[2*i]+dW_2_n[2*i+1] for i in range(N)])
        scheme_paths_1_n= scheme.paths_euler(scheme_type, dW_1_n)
        scheme_paths_2_n = scheme.paths_euler(scheme_type, dW_2_n)
        scheme_paths_1_10n= scheme.paths_euler(scheme_type, dW_1_10n)
        scheme_paths_2_10n = scheme.paths_euler(scheme_type, dW_2_10n)
        error_n = scheme_paths_1_n - np.array([scheme_paths_2_n[2*i] for i in range(N + 1)])
        error_10n = scheme_paths_1_10n - np.array([scheme_paths_2_10n[2*i] for i in range(10*N + 1)])
        E_n = np.max(np.abs(error_n), axis = 0)
        I_M_n = np.mean(E_n)
        E_10n = np.max(np.abs(error_10n), axis = 0)
        I_M_10n = np.mean(E_10n)
        result.append(np.log10(I_M_n) - np.log10(I_M_10n))
    return np.array(result)



def weak_error(f, scheme, scheme_type, steps, M = 10000):
    """ Computes the weak error for the chosen scheme type, including the extrapolation result

    Args : 
        f : test function
        scheme :  instance of CIR class with chosen parameters
        scheme_type (string) : from the 5 types of CIR schemes 
        steps (list<int>) : list of discretization steps
        M (int) : number of simulation trajectories

    Returns :
        numpy.array : array of the weak errors of the chosen 
                        scheme_type for every step
    """
    result = []
    for N in steps:
        dW_2 = np.sqrt(scheme.T / (2*N)) * rng.standard_normal((2*N, M))
        dW_1 = dW_2[::2] + dW_2[1::2]
        paths_1 = scheme.paths_euler(scheme_type, dW_1)
        paths_2 = scheme.paths_euler(scheme_type, dW_2)
        sample_1 = f(paths_1[-1])
        sample_2 = f(paths_2[-1])
        mean_normal = sample_1.mean()
        mean_double = sample_2.mean()
        mean_extrapolation = 2*mean_double - mean_normal

        result.append([mean_normal, mean_double, mean_extrapolation])
    
    return np.array(result)
        


