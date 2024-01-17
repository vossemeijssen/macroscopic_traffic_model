import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from godunovfunctions import *
from evolutionaryfunctions import *
import tqdm


# ------- 1) Get correct data from csv -------
# Define paths
datafolder = os.path.join(os.getcwd(), "data", "a13_2_months")
datafolder_intensityspeed = os.path.join(datafolder, "intensiteit-snelheid-export")
file1_path = os.path.join(datafolder_intensityspeed, "intensiteit-snelheid-export.csv")

# Create pd df
print("loading data")
file1 = pd.read_csv(file1_path)
print("data loaded")
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


# ------- 2) Fit Smulders on q and u using an EA-------
# Define a fitness function
def fitness(u0, qj, qc):
    FR = Smulders(u0, qj, qc)
    u_pred = FR.u(q)
    if qc > qj or u0 < 0 or qj < 0 or qc < 0:
        return -np.linalg.norm(u_pred - u) * 10
    else:
        return -np.linalg.norm(u_pred - u)

# Define parameter for each individual
ind_parameters = {'lower_bound': 10,
                  'upper_bound': 200,
                  'number_of_genes': 3}

# Define parameter for the entire population
pop_parameters = {'n_parents': 2,
                  'offspring_size':(10, ind_parameters['number_of_genes']),
                  'mutation_mean': 0,
                  'mutation_sd': 10,
                  'size': 20}

# Define initial population as Smulders would have wanted
ind_values = [[110, 110, 27]] * pop_parameters["size"]

# Instantiate an evolution
evo = Evolution(pop_parameters, ind_parameters, fitness, ind_values)
# Repeat evolution step 200 epochs
epochs = 20
# Record fitness history
history = []
u0_history = []
qj_history = []
qc_history = []
for _ in tqdm.tqdm(range(epochs)):
    # print('Epoch {}/{}, Progress: {}%\r'.format(_+1, epochs, np.round(((_+1)/epochs)*100, 2)), end="")
    evo.step()
    history.append(evo._best_score)
    u0_history.append(evo._best_individual[0][0])
    qj_history.append(evo._best_individual[0][1])
    qc_history.append(evo._best_individual[0][2])

print('\nResults:')
print('Best individual:', evo.solution.best_individual)
print('Fitness value of best individual:', evo.solution.best_score)

best_FR = Smulders(*evo.solution.best_individual)
plt.plot(q, u, ".", alpha=0.3)
q_test = np.linspace(min(q), max(q), 100)
plt.plot(q_test, best_FR.u(q_test))
plt.xlabel("Density (cars / km)")
plt.ylabel("Average speed (km / hr)")
plt.show()


