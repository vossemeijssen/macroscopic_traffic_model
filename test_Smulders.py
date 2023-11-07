from godunovfunctions import *
import matplotlib.pyplot as plt

q = np.linspace(0, 100, 101)

# Smulders chose for a Dutch motorway: u0 = 110km/h, kc = 27veh/km, and kj = 110veh/km
fr = Smulders(u0=110, qc=80, qj = 110)
f = fr.f(10)
u = fr.u(10)
f = fr.f(q)
u = fr.u(q)

# plt.plot(q, f)
plt.plot(q, f)
plt.show()

