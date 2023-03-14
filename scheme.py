import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng, SeedSequence
import scipy.stats as sps
import pandas as pd
sq = SeedSequence()
rng = default_rng(sq)

class CIR:
    def __init__(self, x0, a, k, sigma, T, lambdaa = 0):
        self.x0 = x0
        self.a = a
        self.k = k
        self.sigma = sigma
        self.T = T   
        self.lambdaa = lambdaa  
   
    def paths_euler_implicit_3(self, dW):
        N, M = dW.shape
        h = self.T / N
        x = np.empty(shape=(N+1,M))
        x[0] = self.x0
        for n in range(1, N+1): 
            x[n] = ((self.sigma*dW[n-1] + 
                 np.sqrt(self.sigma**2 * dW[n-1]**2 +
                   4 * (x[n-1] + (self.a - 0.5*self.sigma**2)*
                        h) * (1 + self.k*h)))/
                        (2*(1+self.k*h))) ** 2
        return x
    
    def paths_euler_implicit_4(self, dW):
        N, M = dW.shape
        h = self.T / N
        x = np.empty(shape=(N+1,M))
        x[0] = self.x0
        for n in range(1, N+1): 
            x[n] = ((0.5*self.sigma*dW[n-1] + np.sqrt(x[n-1])
                  + np.sqrt((0.5*self.sigma * dW[n-1]
                  + np.sqrt(x[n-1]))**2 + 
                   4 * ((self.a*0.5 - 0.125*self.sigma**2)*
                        h) * (1 + self.k*h)))/
                        (2*(1+self.k*h))) ** 2
        return x
    
    def paths_euler_lambdaa(self, dW):
        N, M = dW.shape
        h = self.T / N
        x = np.empty(shape=(N+1,M))
        x[0] = self.x0
        for n in range(1, N+1): 
            x[n] = ((1 - 0.5*self.k*h)*np.sqrt(x[n-1]) + 0.5*self.sigma*dW[n-1]/(1 - 0.5*self.k*h))**2 + (self.a - 0.25*self.sigma**2)*h + self.lambdaa*(dW[n-1]**2 - h)
        return x
    def paths_euler_lambdaa_0(self, dW):
        N, M = dW.shape
        h = self.T / N
        x = np.empty(shape=(N+1,M))
        x[0] = self.x0
        for n in range(1, N+1): 
            x[n] = ((1 - 0.5*self.k*h)*np.sqrt(x[n-1]) + 0.5*self.sigma*dW[n-1]/(1 - 0.5*self.k*h))**2 + (self.a - 0.25*self.sigma**2)*h 
        return x