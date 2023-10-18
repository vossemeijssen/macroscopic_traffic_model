from godunovfunctions import *
from tqdm import tqdm

# Settings
dt = 0.001  # Also known as k
xmin = -1
xmax = 1
xlen = 2001
T = 2
periodic_BC = False

# Set up variables
# x = np.arange(
#     start=  -1,
#     stop=   1+dx,
#     step=   dx,
# )
# x = np.linspace(start=xmin, stop=xmax, num=xlen)
# dx = (xmax - xmin) / (xlen - 1)  # Also known as h

RL = RoadLayout.from_linspace(xmin, xmax, xlen)

q = np.zeros_like(RL.x)

# Set initial values
q[: int(len(q) / 2)] = 0.25
q[int(len(q) / 2) :] = 0.5
# q = 0.25 + 0.25 * np.exp(-0.01 * x**2)  # "Formation of a traffic jam"

# Choose FR
fr = Linear()

# Time loop
GS = GodunovScheme(RL, q, fr, periodic_BC)
timesteps = int(T / dt)
for timestep in tqdm(range(timesteps)):
    GS.time_step(dt)
GS.plotdensity(args="--")



############# PART 2 ###############



RL = RoadLayout.from_linspace(xmin, xmax, 201)

q = np.zeros_like(RL.x)

# Set initial values
q[: int(len(q) / 2)] = 0.25
q[int(len(q) / 2) :] = 0.5
# q = 0.25 + 0.25 * np.exp(-0.01 * x**2)  # "Formation of a traffic jam"

# Choose FR
fr = Linear()

# Time loop
GS = GodunovScheme(RL, q, fr, periodic_BC)
timesteps = int(T / dt)
for timestep in tqdm(range(timesteps)):
    GS.time_step(dt)
GS.plotdensity(args="--")



def correctfunction(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i]<0.5:
            y[i] = 0.25
        elif x[i]>=0.5:
            y[i] = 0.5
    return y
plt.plot(RL.x, correctfunction(RL.x))
plt.legend(["Godunov scheme h=0.001", "Godunov scheme h=0.01", "Exact solution"])
plt.show()