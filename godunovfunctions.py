import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import relu, mse_loss
import torch
# import tqdm


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
    def __init__(self, u0=110., qj=140., qc=30.) -> None:
        self.u0 = u0
        self.qj = qj
        self.qc = qc
        self.gamma = u0 * qc
        if qj != -1:
            self.find_max()
        pass

    def single_u(self, q):
        if q <= self.qc:
            return self.u0 * (1 - q/self.qj)
        else:
            if q == 0:
                return 0
            return self.u0 * self.qc * (1/q - 1/self.qj)

    def u(self, q):
        if isinstance(q, np.ndarray):
            u = np.zeros_like(q)
            for i in range(len(u)):
                u[i] = self.single_u(q[i])
            return u
        return self.single_u(q)
    
    def f(self, q):
        u = self.u(q)
        return q * u

    def f_der(self, q):
        # TODO Define this derivative
        value = 1
        return value
    
    def find_max(self):
        # max of q*u0*(1-q/qj) is q=0.5*qj:
        if self.qc > 0.5 * self.qj:
            self.q_max = 0.5 * self.qj
            self.f_max = self.f(self.q_max)
        else:
            self.q_max = self.qc
            self.f_max = self.f(self.q_max)

    def fit(self, q, f, epochs, lr):
        u0 = torch.tensor([[float(self.u0)]], requires_grad=True)
        qj = torch.tensor([[float(self.qj)]], requires_grad=True)
        qc = torch.tensor([[float(self.qc)]], requires_grad=True)

        qs = torch.tensor(q, requires_grad=False) 
        fs = torch.tensor(f, requires_grad=False)
        us = torch.tensor(f / q, requires_grad=False)

        history = []

        optimizer = torch.optim.SGD([u0, qc, qj], lr=lr)
        for _ in range(epochs):
            
            f_pred = u0 / qj * (qc * qj - qc * qc + (qc + qs - qj) * relu(qc - qs) - qc * relu(qs - qc) )
            loss = mse_loss(f_pred, fs)
            # u_pred = f_pred / qs
            # loss = mse_loss(u_pred, us)

            history.append(float(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            u0.grad.zero_()
            qc.grad.zero_()
            qj.grad.zero_()
        
        self.u0 = float(u0)
        self.qj = float(qj)
        self.qc = float(qc)

        return history


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


def find_road_situation(hectometer, measure_time, MSI_df, max_speed=100, num_lanes=6):
    lanedata = np.zeros(num_lanes)
    for lane_nr in range(1, num_lanes+1):
        # Find closest location
        lane_df = MSI_df[MSI_df["Rijstrook"] == lane_nr]
        if lane_df.empty:
            lanedata[lane_nr - 1] = 0
            continue
        hm_points = lane_df.Hectometrering.unique()
        closest_measuring_location = max(hm_points[hm_points <= hectometer])
        # Only look at the closest location
        lane_df = lane_df[lane_df.Hectometrering == closest_measuring_location]
        # Find the latest update
        latest_update_time = max(lane_df[lane_df.time <= measure_time].time)
        beeldstand = lane_df[lane_df.time == latest_update_time]["Beeldstand"].values[0]
        # Update lanedata according to beeldstand
        if beeldstand in ["blank", "lane_closed_ahead merge_left", "lane_closed_ahead merge_right", "restriction_end"]:
            lanedata[lane_nr - 1] = max_speed
        elif beeldstand in ["lane_closed"]:
            lanedata[lane_nr - 1] = 0
        elif beeldstand.startswith("speedlimit"):
            lanedata[lane_nr - 1] = int(beeldstand.split(" ")[-1])
        else:
            raise KeyError(f"Beeldstand {beeldstand} is not known.")
    return(lanedata)
