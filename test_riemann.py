from godunovfunctions import *
from tqdm import tqdm

# Set up road layout
RL1 = RoadLayout.from_linspace(
    start=-1, 
    stop=1, 
    num=1000
    )

# Define initial condition
def q0(x):
    if x < 0:
        return 0.75
    else:
        return 0.25

fr = Linear()

# Setup Godunov scheme
godunov_scheme = GodunovScheme(road_layout=RL1,
                               q=np.array([q0(xi) for xi in RL1.x]),
                               fr=fr,
                               periodic_BC=False
                               )

# Perform time steps
dt = 0.001
timesteps = int(1.0 / dt)
# for timestep in tqdm(range(timesteps)):
#     godunov_scheme.time_step(dt)

# plot
godunov_scheme.plotdensity()
plt.show()

