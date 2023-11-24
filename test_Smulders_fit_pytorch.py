import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from godunovfunctions import *


# ------- 1) Get correct data from csv -------
# Define paths
datafolder = os.path.join(os.getcwd(), "data", "short_highway")
datafolder_intensityspeed = os.path.join(datafolder, "intensiteit-snelheid-export")
file1_path = os.path.join(datafolder_intensityspeed, "intensiteit-snelheid-export.csv")

# Create pd df
file1 = pd.read_csv(file1_path)
file1["gem_dichtheid"] = file1["gem_intensiteit"] / file1["gem_snelheid"]

# Filter the relevant rows from the raw data
file1_1_location = file1[file1.id_meetlocatie == "RWS01_MONIBAS_0131hrl0117ra"]
filtered = file1_1_location[file1_1_location.voertuigcategorie == "anyVehicle"]
filtered = filtered[filtered.rijstrook_rijbaan == "lane1"]

filtered = filtered[filtered.gem_snelheid != -1]
filtered = filtered[filtered.technical_exclusion != "v"]    
# filtered = filtered[filtered.gem_snelheid <= 150]

filtered["start"] = pd.to_datetime(filtered.start_meetperiode)
mask = (filtered.start.dt.time >= pd.to_datetime("7:00").time()) & (filtered.start.dt.time < pd.to_datetime("19:00").time())
mask2 = (filtered.start.dt.weekday != 5) & (filtered.start.dt.weekday != 6)
filtered = filtered[mask & mask2]

q = filtered["gem_dichtheid"].values.reshape(-1, 1)
u = filtered["gem_snelheid"].values.reshape(-1, 1)
f = q * u

# ------- 2) Fit Smulders on q and u using an EA-------
fr = Smulders()
history = fr.fit(q, f, epochs=1000, lr=0.001)

print(fr.u0, fr.qc, fr.qj)
print(history[-1])
plt.plot(history)
plt.yscale("log")
plt.show()

