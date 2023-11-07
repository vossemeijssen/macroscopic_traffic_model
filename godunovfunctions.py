import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# functions
# Basic Fundamental Relation class
class FR:
    def __init__(self) -> None:
        self.q_max = 0
        self.f_max = 0
        pass

    def f(self, q):
        # Should be able to handle a np array
        pass

    def f_der(self, q):
        # Should be able to handle a np array
        pass

    def find_max(self):
        # Finds the unique solution u to f'(u) = 0
        pass


# Linear fundamental relation
class Linear(FR):
    def __init__(self) -> None:
        self.qmax = 0.5
        self.fmax = self.f(0.5)

    def f(self, q):
        return q - q * q

    def f_der(self, q):
        return 1 - 2 * q
        

class Smulders(FR):
    def __init__(self, u0=-1, qj=-1, qc=-1) -> None:
        self.u0 = u0
        self.qj = qj
        self.qc = qc
        self.gamma = u0 * qc
        pass

    def u(self, q):
        u = np.zeros_like(q)
        for i in range(len(u)):
            if q[i] <= self.qc:
                u[i] = self.u0 * (1 - q[i]/self.qj)
            else:
                if q[i] == 0:
                    u[i] = 0
                u[i] = self.u0 * self.qc * (1/q[i] - 1/self.qj)
        return u
    
    def f(self, q):
        u = self.u(q)
        return q * u

    def f_der(self, q):
        # TODO Define this derivative
        value = 1
        return value
    
    def find_max(self):
        # TODO Define the function that finds these values
        self.q_max = 0
        self.f_max = 0 


# Data class for x and q (road layout)
class RoadLayout():
    def __init__(self, x) -> None:
        self.x = x
        self.xlen = len(x)
        self.dx = np.zeros_like(x)  #dx[i] is the size of cell x[i]. Boundary cells are "half"
        self.dx[0] = (self.x[1] - self.x[0]) / 2
        self.dx[-1] = (self.x[-1] - self.x[-2]) / 2
        for i in range(1, self.xlen - 1):
            self.dx[i] = (self.x[i+1] - self.x[i-1]) / 2

    @classmethod
    def from_linspace(cls, start, stop, num):
        x = np.linspace(start=start, stop=stop, num=num)
        return cls(x)
    
    @classmethod
    def from_arange(cls, start, stop, step):
        x = np.arange(start=start, stop=stop, step=step)
        return cls(x)


# Godunov Scheme class, which handles the time steps
class GodunovScheme():
    def __init__(
            self, 
            road_layout: RoadLayout, 
            q: np.array, 
            fr: FR, 
            periodic_BC=False) -> None:
        self.road_layout = road_layout
        self.q = q
        self.fr = fr
        self.periodic_BC = periodic_BC  # if false: constant BC
    
    def time_step(self, dt):
        f = self.fr.f(self.q)
        f_der = self.fr.f_der(self.q)
        xlen = self.road_layout.xlen

        # Find all q* (or actually, find all f(q*))
        f_q_star = np.zeros_like(self.q)
        for i in range(xlen):
        # When looking at q*[i], we need info from q[i] and q[i]+1
        # Edge case at i_max: assume periodic BC
        # Four cases:
            if f_der[i] >= 0 and f_der[(i + 1) % xlen] >= 0:
                f_q_star[i] = f[i]
            elif f_der[i] < 0 and f_der[(i + 1) % xlen] < 0:
                f_q_star[i] = f[(i + 1) % xlen]
            elif f_der[i] >= 0 and f_der[(i + 1) % xlen] < 0:
                s = (f[(i + 1) % xlen] - f[i]) / (self.q[(i + 1) % xlen] - self.q[i])
                if s >= 0:
                    f_q_star[i] = f[i]
                else:
                    f_q_star[i] = f[(i + 1) % xlen]
            elif f_der[i] < 0 and f_der[(i + 1) % xlen] >= 0:
                f_q_star[i] = self.fr.fmax            
        
        # Now that q* is known, we can calculate the new q values
        new_q = np.zeros_like(self.q)
        if self.periodic_BC:
            # Periodic BC means q[imax] = q[0], they are the same point.
            for i in range(1, xlen):
                new_q[i] = self.q[i] - dt / self.road_layout.dx[i] * (f_q_star[i] - f_q_star[i - 1])
            new_q[0] = new_q[-1]
        else:
            new_q[0] = self.q[0]
            new_q[-1] = self.q[-1]
            for i in range(1, xlen - 1):
                new_q[i] = self.q[i] - dt / self.road_layout.dx[i] * (f_q_star[i] - f_q_star[i - 1])

        self.q = new_q


    def plotdensity(self, args=""):
        plt.plot(self.road_layout.x, self.q, args)
        plt.xlabel("Length of the road")
        plt.ylabel("Density")

