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


# Plot functions
def plot_density(x, q):
    assert len(q) == len(x)
    plt.plot(x, q, "-.")


