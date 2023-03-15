import numpy as np
from scipy.integrate import quad


class Heston:
    def __init__(self, kappa, theta, sigma, rho, v0,r,s0,T):
        self.kappa =kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.s0 = s0
        self.T = T


    def call(self, K):
        p1, p2 = self.integral(K)
        return self.s0*p1 - K*np.exp(-self.r * self.T) * p2 

    def integral(self,K):

        integrand_1 =  lambda phi: (np.exp(-1j * phi * np.log(K)) * self.f_1(phi) / (1J*phi) ).real
        integrand_2 = lambda phi: (np.exp(-1j * phi * np.log(K)) * self.f_2(phi) / (1J*phi) ).real
        return  (0.5 + (1 / np.pi) * quad(integrand_1, 0, 100)[0]), (0.5 + (1 / np.pi) * quad(integrand_2, 0, 100)[0]) 

    def f_1(self, phi):
        
        u = 0.5
        b = self.kappa - self.rho * self.sigma
        a = self.kappa * self.theta
        x = np.log(self.s0)
        d = np.sqrt((self.rho * self.sigma * phi * 1j - b)**2 - self.sigma**2 * (2 * u * phi * 1j - phi**2))
        g = (b - self.rho * self.sigma * phi * 1j + d) / (b - self.rho * self.sigma * phi * 1j - d)
        C = self.r * phi * 1j * self.T + (a / self.sigma**2)*((b - self.rho * self.sigma * phi * 1j + d) * self.T - 2 * np.log((1 - g * np.exp(d * self.T))/(1 - g)))
        D = (b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2 * ((1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T)))
        return np.exp(C + D * self.v0 + 1j * phi * x)
    
    def f_2(self, phi):
        u = -0.5
        b = self.kappa - self.rho * self.sigma
        a = self.kappa * self.theta
        x = np.log(self.s0)
        d = np.sqrt((self.rho * self.sigma * phi * 1j - b)**2 - self.sigma**2 * (2 * u * phi * 1j - phi**2))
        g = (b - self.rho * self.sigma * phi * 1j + d) / (b - self.rho * self.sigma * phi * 1j - d)
        C = self.r * phi * 1j * self.T + (a / self.sigma**2)*((b - self.rho * self.sigma * phi * 1j + d) * self.T - 2 * np.log((1 - g * np.exp(d * self.T))/(1 - g)))
        D = (b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2 * ((1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T)))
        return np.exp(C + D * self.v0 + 1j * phi * x)



