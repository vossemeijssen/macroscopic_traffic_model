from godunovfunctions import *
from tqdm import tqdm

# Settings
dt = 0.01  # Also known as k
xmin = -30
xmax = 30
xlen = 5000
T = 40
periodic_BC = False

# Set up variables
# x = np.arange(
#     start=  -1,
#     stop=   1+dx,
#     step=   dx,
# )
x = np.linspace(start=xmin, stop=xmax, num=xlen)
dx = (xmax - xmin) / (xlen - 1)  # Also known as h

q = np.zeros_like(x)

# Set initial values
q[: int(len(q) / 2)] = 0.5
q[int(len(q) / 2) :] = 0.25
# q = 0.25 + 0.25 * np.exp(-0.01 * x**2)  # "Formation of a traffic jam"

# Choose FR
fr = Linear()

# Time loop
timesteps = int(T / dt)
for timestep in tqdm(range(timesteps)):
    f = fr.f(q)
    f_der = fr.f_der(q)

    # Find all q* (or actually, find all f(q*))
    f_q_star = np.zeros_like(q)
    for i in range(xlen):
        # When looking at q*[i], we need info from q[i] and q[i]+1
        # Edge case at i_max: in this case, q[i_max] will be constant
        # Four cases:
        if f_der[i] >= 0 and f_der[(i + 1) % xlen] >= 0:
            f_q_star[i] = f[i]
        elif f_der[i] < 0 and f_der[(i + 1) % xlen] < 0:
            f_q_star[i] = f[(i + 1) % xlen]
        elif f_der[i] >= 0 and f_der[(i + 1) % xlen] < 0:
            s = (f[(i + 1) % xlen] - f[i]) / (q[(i + 1) % xlen] - q[i])
            if s >= 0:
                f_q_star[i] = f[i]
            else:
                f_q_star[i] = f[(i + 1) % xlen]
        elif f_der[i] < 0 and f_der[(i + 1) % xlen] >= 0:
            f_q_star[i] = fr.fmax

    # Now that q* is known, we can calculate the new q values
    new_q = np.zeros_like(q)
    if periodic_BC:
        for i in range(xlen):
            new_q[i] = q[i] - dt / dx * (f_q_star[i] - f_q_star[(i - 1) % xlen])
    else:
        new_q[0] = q[0]
        new_q[-1] = q[-1]
        for i in range(1, xlen - 1):
            new_q[i] = q[i] - dt / dx * (f_q_star[i] - f_q_star[i - 1])

    q = new_q

plot_density(x, q)

def correctfunction(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i]<-0.5:
            y[i] = 0.75
        elif x[i]>0.5:
            y[i] = 0.25
        else:
            y[i] = 0.5*(1-x[i])
    return y
plt.plot(x, correctfunction(x))



plt.legend(["Godunov scheme h=0.001", "Godunov scheme h=0.01", "Exact solution"])
plt.show()