import numpy as np
from scipy.integrate import quad


class Heston:
    def __init__(self, k, theta, sigma, rho, v0,r,s0,T, lambdaa = 0):
        self.k =k
        self.theta = theta
        self.a = self.theta*self.k
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.s0 = s0
        self.r = r
        self.T = T
        self.lambdaa = lambdaa

    def paths_euler_implicit_3(self, dW1,dW2):
        N, M = dW1.shape
        h = self.T / N
        vol_square = np.empty(shape=(N+1,M))
        s = np.empty(shape=(N+1,M))
        s[0] = self.s0
        vol_square[0] = self.v0
        dW_s = self.rho * dW1 + np.sqrt((1 - self.rho**2))*dW2

        for n in range(1, N+1): 
            det = self.sigma**2 * (dW1[n-1])**2 + 4 * (vol_square[n-1] + (self.a - 0.5*self.sigma**2)*h) * (1 + self.k*h)
            val = ((self.sigma*dW1[n-1] + np.sqrt(self.sigma**2 * (dW1[n-1])**2 + 4 * (vol_square[n-1] + (self.a - 0.5*self.sigma**2)*h) * (1 + self.k*h)))/(2*(1+self.k*h))) ** 2
            vol_square[n][det>=0] = val[det>=0]

        vol = np.sqrt(vol_square[:-1])
        for n in range(1, N+1):
            s[n] = s[n-1] + self.r * s[n-1] * h + vol[n-1] * s[n-1] * dW_s[n-1]

        return s, vol

    def paths_euler_implicit_4(self, dW1,dW2):

        N, M = dW1.shape
        h = self.T / N
        vol_square = np.empty(shape=(N+1,M))
        s = np.empty(shape=(N+1,M))
        s[0] = self.s0
        vol_square[0] = self.v0
        dW_s = self.rho * dW1 + np.sqrt((1 - self.rho**2))*dW2

        for n in range(1, N+1): 
            det = (0.5*self.sigma * dW1[n-1] + np.sqrt(vol_square[n-1]))**2 + 4 * ((self.a*0.5 - 0.125*self.sigma**2)*h) * (1 + 0.5*self.k*h)
            val = ((0.5*self.sigma*dW1[n-1] + np.sqrt(vol_square[n-1]) + np.sqrt((0.5*self.sigma * dW1[n-1] + np.sqrt(vol_square[n-1]))**2 + 4 * ((self.a*0.5 - 0.125*self.sigma**2)*h) * (1 + 0.5*self.k*h)))/(2*(1+ 0.5*self.k*h))) ** 2
            vol_square[n][det>=0] = val[det>=0]

        vol = np.sqrt(vol_square[:-1])
        for n in range(1, N+1):
            s[n] = s[n-1] + self.r * s[n-1] * h + vol[n-1] * s[n-1] * dW_s[n-1]

        return s,vol


    def paths_euler_lambdaa(self, dW1,dW2):
        N, M = dW1.shape
        h = self.T / N
        vol_square = np.empty(shape=(N+1,M))
        s = np.empty(shape=(N+1,M))
        s[0] = self.s0
        vol_square[0] = self.v0
        dW_s = self.rho * dW1 + np.sqrt((1 - self.rho**2))*dW2
        for n in range(1, N+1): 
            vol_square[n] = ((1 - 0.5*self.k*h)*np.sqrt(vol_square[n-1]) + 0.5*self.sigma*dW1[n-1]/(1 - 0.5*self.k*h))**2 + (self.a - 0.25*self.sigma**2)*h + self.lambdaa*(dW1[n-1]**2 - h)
            vol_square[n][vol_square[n] < 0] = 0          
       
        vol = np.sqrt(vol_square[:-1])


        for n in range(1, N+1):
            s[n] = s[n-1] + self.r * s[n-1] * h + vol[n-1] * s[n-1] * dW_s[n-1]

        return s, vol

    def paths_euler_lambdaa_0(self, dW1,dW2):
        N, M = dW1.shape
        h = self.T / N
        vol_square = np.empty(shape=(N+1,M))
        s = np.empty(shape=(N+1,M))
        s[0] = self.s0
        vol_square[0] = self.v0
        dW_s = self.rho * dW1 + np.sqrt((1 - self.rho**2))*dW2
        for n in range(1, N+1): 
            vol_square[n] = ((1 - 0.5*self.k*h)*np.sqrt(vol_square[n-1]) + 0.5*self.sigma*dW1[n-1]/(1 - 0.5*self.k*h))**2 + (self.a - 0.25*self.sigma**2)*h 
            vol_square[n][vol_square[n] < 0] = 0
       
        vol = np.sqrt(vol_square[:-1])


        for n in range(1, N+1):
            s[n] = s[n-1] + self.r * s[n-1] * h + vol[n-1] * s[n-1] * dW_s[n-1]

        return s, vol

    def paths_euler_dd(self, dW1, dW2):
        
        N, M = dW1.shape
        h = self.T / N
        vol_square = np.empty(shape=(N+1,M))
        s = np.empty(shape=(N+1,M))
        s[0] = self.s0
        vol_square[0] = self.v0
        dW_s = self.rho * dW1 + np.sqrt((1 - self.rho**2))*dW2
        for n in range(1, N+1): 
            vol_square[n] = vol_square[n-1] + h*(self.a - self.k* vol_square[n-1]) + self.sigma * np.sqrt(vol_square[n-1] * (vol_square[n-1] >0)) * dW1[n-1]
       
        vol = np.sqrt(vol_square[:-1])


        for n in range(1, N+1):
            s[n] = s[n-1] + self.r * s[n-1] * h + vol[n-1] * s[n-1] * dW_s[n-1]

        return s, vol


    def paths_euler_diop(self, dW1,dW2):

        N, M = dW1.shape
        h = self.T / N
        vol_square = np.empty(shape=(N+1,M))
        s = np.empty(shape=(N+1,M))
        s[0] = self.s0
        vol_square[0] = self.v0
        dW_s = self.rho * dW1 + np.sqrt((1 - self.rho**2))*dW2
        for n in range(1, N+1): 
            vol_square[n] = vol_square[n-1] + h*(self.a - self.k* vol_square[n-1]) + self.sigma * np.sqrt(vol_square[n-1]) * dW1[n-1]
            vol_square[n] = np.abs(vol_square[n])       
        vol = np.sqrt(vol_square[:-1])
        for n in range(1, N+1):
            s[n] = s[n-1] + self.r * s[n-1] * h + vol[n-1] * s[n-1] * dW_s[n-1]

        return s, vol

    def paths_euler(self, scheme_type, dW1, dW2):
        """ Computes a diffusion the chosen scheme type with the size
            of the brownian matrice to simulate the stock price in the heston model

        Args : 
            scheme_type (string) : from the 5 types of CIR schemes 
            dW1 (numpy.array) : simulation of brownians of size (N+1,M)
            dW2 (numpy.array) : simulation of brownians of size (N+1,M)

        Returns :
            numpy.array : array of diffusion of the chosen scheme_type of
                            size (N+1)
        """
        match scheme_type:
            case "implicit_3":
                return self.paths_euler_implicit_3(dW1,dW2)
            case "implicit_4":
                return self.paths_euler_implicit_4(dW1,dW2)
            case "E_lambda":
                return self.paths_euler_lambdaa(dW1,dW2)
            case "E_0":
                return self.paths_euler_lambdaa_0(dW1,dW2)
            case "D-D":
                return self.paths_euler_dd(dW1,dW2)
            case "Diop":
                return self.paths_euler_diop(dW1,dW2)
    


    def call(self, K):

        """
        compute the price of Call by using the semi-closed formula

        Args:
            K (float): the strik of the option  
        """
        p1, p2 = self.integral(K)
        return self.s0*p1 - K*np.exp(-self.r * self.T) * p2 

    def integral(self,K):

        integrand_1 =  lambda phi: (np.exp(-1j * phi * np.log(K)) * self.f_1(phi) / (1J*phi) ).real
        integrand_2 = lambda phi: (np.exp(-1j * phi * np.log(K)) * self.f_2(phi) / (1J*phi) ).real
        return  (0.5 + (1 / np.pi) * quad(integrand_1, 0, 100)[0]), (0.5 + (1 / np.pi) * quad(integrand_2, 0, 100)[0]) 

    def f_1(self, phi):

        
        u = 0.5
        b = self.k - self.rho * self.sigma
        a = self.k * self.theta
        x = np.log(self.s0)
        d = np.sqrt((self.rho * self.sigma * phi * 1j - b)**2 - self.sigma**2 * (2 * u * phi * 1j - phi**2))
        g = (b - self.rho * self.sigma * phi * 1j + d) / (b - self.rho * self.sigma * phi * 1j - d)
        C = self.r * phi * 1j * self.T + (a / self.sigma**2)*((b - self.rho * self.sigma * phi * 1j + d) * self.T - 2 * np.log((1 - g * np.exp(d * self.T))/(1 - g)))
        D = (b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2 * ((1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T)))
        return np.exp(C + D * self.v0 + 1j * phi * x)
    
    def f_2(self, phi):
        u = -0.5
        b = self.k
        a = self.k * self.theta
        x = np.log(self.s0)
        d = np.sqrt((self.rho * self.sigma * phi * 1j - b)**2 - self.sigma**2 * (2 * u * phi * 1j - phi**2))
        g = (b - self.rho * self.sigma * phi * 1j + d) / (b - self.rho * self.sigma * phi * 1j - d)
        C = self.r * phi * 1j * self.T + (a / self.sigma**2)*((b - self.rho * self.sigma * phi * 1j + d) * self.T - 2 * np.log((1 - g * np.exp(d * self.T))/(1 - g)))
        D = (b - self.rho * self.sigma * phi * 1j + d) / self.sigma**2 * ((1 - np.exp(d * self.T)) / (1 - g * np.exp(d * self.T)))
        return np.exp(C + D * self.v0 + 1j * phi * x)



