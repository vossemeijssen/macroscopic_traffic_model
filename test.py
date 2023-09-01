from gunodov_functions import *
from tqdm import tqdm

# Settings
dt = 0.001 # Also known as k
dx = 0.001 # Also known as h
T = 1

# Set up variables
x = np.arange(
    start=  -1,
    stop=   1+dx,
    step=   dx,
)
q = np.zeros_like(x)

# Set initial values
q[:int(len(q)/2)] = 0.75
q[int(len(q)/2):] = 0.25

# Choose FR
fr = Linear()

# Time loop
timesteps = int(T / dt)
for timestep in tqdm(range(timesteps)):
    f = fr.f(q)
    f_der = fr.f_der(q)

    # Find all q* (or actually, find all f(q*))
    f_q_star = np.zeros_like(q)
    for i in range(len(f_q_star)-1):
        # When looking at q*[i], we need info from q[i] and q[i]+1
        # Edge case at i_max: in this case, q[i_max] will be constant
        # Four cases:
        if f_der[i] >= 0 and f_der[i+1] >= 0:
            f_q_star[i] = f[i]
        elif f_der[i] < 0 and f_der[i+1] < 0:
            f_q_star[i] = f[i+1]
        elif f_der[i] >= 0 and f_der[i+1] < 0:
            s = (f[i+1] - f[i])/(q[i+1] - q[i])
            if s >= 0:
                f_q_star[i] = f[i]
            else:
                f_q_star[i] = f[i+1]
        elif f_der[i] < 0 and f_der[i+1] >= 0:
            f_q_star[i] = fr.fmax

    # Now that q* is known, we can calculate the new q values
    new_q = np.zeros_like(q)
    new_q[0] = q[0]
    new_q[-1] = q[-1]
    for i in range(1, len(q)-1):
        new_q[i] = q[i] - dt / dx * (f_q_star[i] - f_q_star[i-1])

    q = new_q

plot_density(x, q)

