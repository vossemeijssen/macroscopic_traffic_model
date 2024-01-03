from godunovfunctions import *
from tqdm import tqdm

# Settings
dt = 0.005  # Also known as k
xmin = -30 
xmax = 30
xlen = 500
T = 40
periodic_BC = True

RL = RoadLayout.from_linspace(xmin, xmax, xlen)

q = np.zeros_like(RL.x)

# Set initial values
q = 0.25 + 0.25 * np.exp(-0.01 * RL.x**2)  # "Formation of a traffic jam"

# Choose FR
# fr = Linear()
fr = Smulders(u0=1, qj=1, qc=1)

# Time loop
GS = GodunovScheme(RL, q, fr, periodic_BC)
timesteps = int(T / dt)
for timestep in tqdm(range(timesteps)):
    GS.time_step(dt)

GS.plotdensity()
plt.show()