import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import relu, mse_loss
import torch.optim as optim
import os
import tqdm
from sklearn.model_selection import train_test_split


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
        self.q_max = 0.5
        self.f_max = self.f(0.5)

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
        if q > self.qj:
            return 0
        elif q <= 0:
            return self.u0
        elif q <= self.qc:
            return self.u0 * (1 - q/self.qj)
        else:
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

    def single_f(self, q):
        u = self.single_u(q)
        return q * u

    def f_der(self, q):
        if isinstance(q, np.ndarray):
            raise NotImplementedError
        if q < self.qc:
            return self.u0 * (1 - 2 * q / self.qj)
        else:
            return self.u0 * self.qc * (1 - q / self.qj)
    
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

        optimizer = optim.SGD([u0, qc, qj], lr=lr)
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
    
    def f_from_ql_qr(self, ql, qr):
        if ql <= self.q_max and qr <= self.q_max:
            return self.single_f(ql)
        if ql > self.q_max and qr > self.q_max:
            return self.single_f(qr)
        if ql <= self.q_max and qr > self.q_max:
            fl = self.single_f(ql)
            fr = self.single_f(qr)
            s = (fl - fr) / (ql - qr)
            if s >= 0:
                return fl
            else:
                return fr
        if ql > self.q_max and qr <= self.q_max:
            return self.f_max


# Data class for x (road layout)
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
            fr: FR | list[FR], 
            periodic_BC=False) -> None:
        self.road_layout = road_layout
        self.q = q
        if isinstance(fr, FR):
            self.fr = [fr] * (len(q)-1)
        else:
            assert len(fr) == len(q)-1
            self.fr = fr
        self.periodic_BC = periodic_BC  # if false: constant BC
    
    def time_step(self, dt):
        f = [self.fr[i].f(self.q[i]) for i in range(len(self.q)-1)]
        f.append(self.fr[-1].f(self.q[-1]))
        f_der = [self.fr[i].f_der(self.q[i]) for i in range(len(self.q)-1)]
        f_der.append(self.fr[-1].f_der(self.q[-1]))
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
                f_q_star[i] = self.fr[i].f_max            
        
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

    def time_step_2(self, dt):
        new_q = np.zeros_like(self.q)
        fluxes = [
            self.fr[i].f_from_ql_qr(self.q[i], self.q[i+1]) 
            for i in range(len(self.fr))
            ]
        
        if self.periodic_BC:
            raise NotImplementedError
        else:
            new_q[0] = self.q[0]
            new_q[-1] = self.q[-1]
            for i in range(1, len(new_q) - 1):
                new_q[i] = self.q[i] - dt / self.road_layout.dx[i] * (fluxes[i] - fluxes[i-1])
        
        self.q = new_q

    def plotdensity(self, args=""):
        plt.plot(self.road_layout.x, self.q, args)
        plt.xlabel("Length of the road")
        plt.ylabel("Density")


def find_road_situation(hectometer, measure_time, MSI_df, direction="R", max_speed=100, num_lanes=6):
    lanedata = np.zeros(num_lanes)
    # Find closest location
    loc_df = MSI_df[MSI_df["DVK"] == direction]
    hm_points = loc_df.Hectometrering.unique()
    if direction == "R": 
        closest_measuring_location = max(hm_points[hm_points <= hectometer])
    else:
        closest_measuring_location = min(hm_points[hm_points >= hectometer])
    # Only look at the closest location
    loc_df = loc_df[loc_df["Hectometrering"] == closest_measuring_location]
    for lane_nr in range(1, num_lanes+1):
        # If there is no lane_nr in the closest location, then there is no lane. Set speed 0.
        lane_df = loc_df[loc_df["Rijstrook"] == lane_nr]
        if lane_df.empty:
            lanedata[lane_nr - 1] = 0
            continue
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


def add_MSI_information(IS_loc_df, MSI_df, hectometer, direction, max_speed=100.0, num_lanes=6):
    # In this function, IS_loc_df is the subset of IS_df with only the data of 1 location. 
    # We will create num_lanes=6 empty columns of return_df, with the same number of rows as loc_df
    return_df = pd.DataFrame()

    # Find closest location, call it "closest_measuring_location"
    dir_df = MSI_df[MSI_df["DVK"] == direction]
    hm_points = dir_df.Hectometrering.unique()
    try:
        if direction == "R": 
            closest_measuring_location = max(hm_points[hm_points <= hectometer])
        else:
            closest_measuring_location = min(hm_points[hm_points >= hectometer])
    except ValueError:
        print(f'ERROR: Matrix data unknown for measuring point {IS_loc_df["id_meetlocatie"].unique()[0]}\nat hectometer {hectometer}, direction {direction}.\nTry adding more MSI data.')
        for lane_nr in range(1, num_lanes+1):
            return_df[lane_nr] = [0] * len(IS_loc_df)
        return return_df
        # raise ValueError(f"Matrix data unknown for measuring point at hectometer {hectometer}.\nTry adding more MSI data.")
    # Only look at the closest location
    loc_df = dir_df[dir_df["Hectometrering"] == closest_measuring_location]
    # print(loc_df)
    # Define MSI configs:
    MSIconfigs = {
        "blank": max_speed,
        "lane_closed_ahead merge_left": max_speed,
        "lane_closed_ahead merge_right": max_speed,
        "restriction_end": max_speed,
        "lane_closed": 0,
        "speedlimit 100": 100,
        "speedlimit 90": 90,
        "speedlimit 80": 80,
        "speedlimit 70": 70,
        "speedlimit 60": 60, 
        "speedlimit 50": 50,
        "speedlimit 30": 30,
        "unknown": max_speed
    }

    for lane_nr in range(1, num_lanes+1):
        # If there is no lane_nr in the closest location, then there is no lane. 
        # Set max speed equal to 0 for the whole time on this lane
        lane_df = loc_df[loc_df["Rijstrook"] == lane_nr]
        if lane_df.empty:
            return_df[lane_nr] = [0] * len(IS_loc_df)
            continue
                
        # Now, for every row in IS_loc_df, find the latest update time
        change_times = lane_df.time
        last_update_time = change_times.iloc[0]
        last_update = MSIconfigs[lane_df[change_times == last_update_time]["Beeldstand"].values[0]]
        next_update_time = change_times.iloc[1]
        column = []
        for measure_time in IS_loc_df["start"]:
            if measure_time < next_update_time:
                column.append(last_update)
            else:
                last_update_time = max(lane_df[change_times <= measure_time].time)
                last_update = MSIconfigs[lane_df[change_times == last_update_time]["Beeldstand"].values[0]]
                next_times = lane_df[change_times > measure_time].time
                if next_times.empty:
                    next_update_time = max(IS_loc_df.start)
                else:
                    next_update_time = min(next_times)
                column.append(last_update)
            # beeldstand = lane_df[change_times == latest_update_time]["Beeldstand"].values[0]
            # column.append(MSIconfigs[beeldstand])
        return_df[lane_nr] = column
    return return_df


def load_data(datafolder, print_logs=False):
    # Define paths
    datafolder_msi = os.path.join(datafolder, "msi-export")
    msi_path = os.path.join(datafolder_msi, "msi-export.csv")
    datafolder_intensityspeed = os.path.join(datafolder, "intensiteit-snelheid-export")
    intensityspeed_path = os.path.join(datafolder_intensityspeed, "intensiteit-snelheid-export.csv")

    # Load data
    if print_logs:
        print("Loading data")
    IS_df = pd.read_csv(intensityspeed_path, low_memory=False)
    MSI_df = pd.read_csv(msi_path, low_memory=False)

    # Add columns
    if print_logs:
        print("Adding columns to IS and MSI data")
    IS_df["gem_dichtheid"] = IS_df["gem_intensiteit"] / IS_df["gem_snelheid"]
    MSI_df["time"] = pd.to_datetime(MSI_df["Datum en tijd beeldstandwijziging"])

    # Filter out the relevant data in the intensity-speed df
    IS_df = IS_df[IS_df.voertuigcategorie == "anyVehicle"]
    IS_df = IS_df[IS_df.gem_snelheid != -1]
    IS_df = IS_df[IS_df.technical_exclusion != "v"]    
    IS_df["start"] = pd.to_datetime(IS_df.start_meetperiode)
    mask = (IS_df.start.dt.time >= pd.to_datetime("7:00").time()) & (IS_df.start.dt.time < pd.to_datetime("19:00").time())
    mask2 = (IS_df.start.dt.weekday != 5) & (IS_df.start.dt.weekday != 6)
    IS_df = IS_df[mask & mask2]

    # Aggregate the lanes together
    filtered = IS_df[["start", "id_meetlocatie"]].drop_duplicates()
    grouped = IS_df.groupby(["start", "id_meetlocatie"])
    tot = grouped["gem_intensiteit"].transform("sum")
    gem_snelheid_weighted = (IS_df["gem_snelheid"] / tot * IS_df["gem_intensiteit"]).groupby([IS_df["start"], IS_df["id_meetlocatie"]]).transform("sum")
    filtered["gem_intensiteit"] = tot
    filtered["gem_snelheid"] = gem_snelheid_weighted
    filtered["gem_dichtheid"] = filtered["gem_intensiteit"] / filtered["gem_snelheid"]
    IS_df = filtered

    # Add direction and hectometer info to all rows of IS_df
    direction_dict = {}
    hectometer_dict = {}
    for i in IS_df.id_meetlocatie.unique():
        if i[6:13] == "MONIBAS":
            if i[14:18] == "0131":
                direction_dict[i] = i[20].upper()
                hectometer_dict[i] = int(i[21:25]) / 10
        else:
            raise NameError(f"Please clarify id {i}")
        
    # Get all MSI information on all IS_df locations    
    IS_df = IS_df[IS_df["id_meetlocatie"] != "RWS01_MONIBAS_0131hrr0119ra"]
    results = pd.DataFrame()
    if print_logs:
        print("Adding MSI information on all locations")
        iteration_unit = tqdm.tqdm(IS_df["id_meetlocatie"].unique())
    else:
        iteration_unit = IS_df["id_meetlocatie"].unique()
    for measuring_location in iteration_unit:
        hectometer = hectometer_dict[measuring_location]
        direction = direction_dict[measuring_location]
        ret = add_MSI_information(IS_df[IS_df["id_meetlocatie"] == measuring_location], MSI_df, hectometer, direction)
        results = pd.concat([results, ret])
    
    # Combine all MSI information on all IS_df locations into df
    df = pd.concat([IS_df.reset_index(), results.reset_index()], axis=1)

    return df


class NeuralNetwork(nn.Module):
    def __init__(self, lin_stack, bias_init_function, weights_init_function, *args, **kwargs) -> None:
        super().__init__()
        self.linear_stack = lin_stack
        for fc1 in self.linear_stack:
            if hasattr(fc1, "bias"):
                bias_init_function(fc1.bias)
            if hasattr(fc1, "weight"):
                weights_init_function(fc1.weight)

    
    def forward(self, x):
        fr_params = self.linear_stack(x[:, :6])
        
        u0 = fr_params[:, 0]
        qj = fr_params[:, 1]
        qc = fr_params[:, 2]

        q = x[:, -1]

        f_pred = u0 / qj * (qc * qj - qc * qc + (qc + q - qj) * F.relu(qc - q) - qc * F.relu(q - qc))
        return F.relu(f_pred)
    
    # def _initialize_parameters(self):
    #     for fc1 in self.linear_stack:
    #         torch.nn.init.zeros_(fc1.bias)
    #         torch.nn.init.xavier_uniform_(fc1.weight)
    
    def get_params(self, x):
        fr_params = self.linear_stack(x[:, :-1])
        return fr_params


