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
        pass

    def f_der(self, q):
        pass

    def find_max(self):
        # Finds the unique solution u to f'(u) = 0
        pass


# Linear fundamental relation
class Linear(FR):
    def __init__(self) -> None:
        self.qmax = 0.5
        self.fmax = self.f(0.5)
        pass

    def f(self, q):
        return q - q * q

    def f_der(self, q):
        return 1 - 2 * q


# Godunov Scheme class, which handles the time steps
class GodunovScheme():
    def __init__(self, x, dx, q, fr: FR, periodic_BC=False) -> None:
        self.x = x
        self.dx = dx
        self.q = q
        self.fr = fr
        self.periodic_BC = periodic_BC  # if false: constant BC
    
    def time_step(self, dt):
        f = self.fr.f(self.q)
        f_der = self.fr.f_der(self.q)
        xlen = len(self.q)

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
            for i in range(xlen):
                new_q[i] = self.q[i] - dt / self.dx * (f_q_star[i] - f_q_star[(i - 1) % xlen])
        else:
            new_q[0] = self.q[0]
            new_q[-1] = self.q[-1]
            for i in range(1, xlen - 1):
                new_q[i] = self.q[i] - dt / self.dx * (f_q_star[i] - f_q_star[i - 1])

        self.q = new_q


    def plotdensity(self, args=""):
        plt.plot(self.x, self.q, args)

