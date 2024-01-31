from godunovfunctions import *
import numpy as n_splits
import matplotlib.pyplot as plt
from tqdm import tqdm

# Settings
dt = 0.001  # Also known as k
xmin = 0
xmax = 2
xlen = 2001
f1 = Smulders(u0=1, qj=1, qc=1)
f2 = Smulders(u0=1, qj=0.5, qc=0.5)
fig = plt.figure(figsize=(10, 6))
# Plot FRs
if False:
    for i, f in enumerate([f1, f2]):
        x = np.linspace(0, f.qj, 100)
        y = f.f(x)
        plt.plot(x, y, label=i+1)
    plt.legend()
    plt.show()

# Setting initial values
x = np.linspace(xmin, xmax, xlen)
q0 = np.zeros_like(x)
q0[0] = 0.5
fr_list = [f1]*int((xlen-1)/2) + [f2]*int((xlen-1)/2)

# Creating objects
RL = RoadLayout(x)
GS = GodunovScheme(
    road_layout=RL,
    q=q0,
    fr=fr_list,
    periodic_BC=False,
)

# Running time loop
fig, axs = plt.subplots(2, 2)

timesteps = int(0.5 / dt)
for timestep in tqdm(range(timesteps)):
    GS.time_step(dt)
GS.plotdensity(ax_object=axs[0, 0])

timesteps = int(1.5 / dt)
for timestep in tqdm(range(timesteps)):
    GS.time_step(dt)
GS.plotdensity(ax_object=axs[0, 1])

timesteps = int(2 / dt)
for timestep in tqdm(range(timesteps)):
    GS.time_step(dt)
GS.plotdensity(ax_object=axs[1, 0])

timesteps = int(4 / dt)
for timestep in tqdm(range(timesteps)):
    GS.time_step(dt)
GS.plotdensity(ax_object=axs[1, 1])

axs[0, 0].legend(["T=0.5"])
axs[0, 1].legend(["T=2"])
axs[1, 0].legend(["T=4"])
axs[1, 1].legend(["T=8"])

plt.show()
