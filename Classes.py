from abc import ABC, abstractmethod
import numpy as np


class Scheme(ABC):

    def generate_brownian_trajectory(self):
        trajectory = []
        for i in range(1, self.n + 1):
            trajectory.append(self.T / self.n *np.random.normal())
        self.brownian_trajectory = trajectory

    def set_scheme_coefficients(self, x0, a, k, sigma) :
        self.x0 = x0
        self.a = a
        self.k = k
        self.sigma = sigma

    def set_time_grid_parameters(self, T, n):
        self.T = T
        self.n = n

 #   def set_time_grid(self, n):
  #      self.n = n
  #      self.grid = [i*self.T/n  for i in range(n+1)]
 
    @abstractmethod
    def simulate(self):
        pass


class Implicit_3(Scheme):
    def __init__(self, x0, a, k, sigma, T, n):
        self.set_scheme_coefficients(x0, a, k, sigma)
        self.set_time_grid_parameters(T, n)
        self.generate_brownian_trajectory()

    def simulate(self):
        diffusion = [self.x0]
        for i in range(1, self.n + 1):
            x = ((self.sigma*self.brownian_trajectory[i-1] + 
                 np.sqrt(self.sigma**2 * self.brownian_trajectory[i-1]**2 +
                   4 * (diffusion[-1] + (self.a - 0.5*self.sigma**2)*
                        self.T / self.n) * (1 + self.k*self.T/self.n)))/
                        (2*(1+self.k*self.T/self.n))) ** 2
            diffusion.append(x)
        self.simulation = diffusion

    def __str__(self):
        return "Implicit number 3 with parameters : \n a : " + str(self.a) + "\n k : " + str(self.k) + "\n sigma : " + str(self.sigma)

class Implicit_4(Scheme):
    def __init__(self, x0, a, k, sigma, T, n):
        self.set_scheme_coefficients(x0, a, k, sigma)
        self.set_time_grid_parameters(T, n)
        self.generate_brownian_trajectory()

    def simulate(self):
        diffusion = [self.x0]
        for i in range(1, self.n + 1):
            x = ((0.5*self.sigma*self.brownian_trajectory[i-1] + np.sqrt(diffusion[-1])
                  + np.sqrt((0.5*self.sigma * self.brownian_trajectory[i-1]
                  + np.sqrt(diffusion[-1]))**2 + 
                   4 * ((self.a*0.5 - 0.125*self.sigma**2)*
                        self.T / self.n) * (1 + self.k*self.T/self.n)))/
                        (2*(1+self.k*self.T/self.n))) ** 2
            diffusion.append(x)
        self.simulation = diffusion

    def __str__(self):
        return "Implicit number 4 with parameters : \n a : " + str(self.a) + "\n k : " + str(self.k) + "\n sigma : " + str(self.sigma)