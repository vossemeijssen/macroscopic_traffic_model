import os
import torch
from torch.nn.functional import relu, mse_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import tqdm
import pandas as pd
import numpy as np
from godunovfunctions import *


# ------- 1) Get correct data from csv -------
print("Downloading and filtering data...")
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

print("Data is ready.")

# ------- 2) Apply pytorch NN -------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

u0 = torch.tensor([[110.]], requires_grad=True)
qc = torch.tensor([[30.]], requires_grad=True)
qj = torch.tensor([[140.]], requires_grad=True)

qs = torch.tensor(q, requires_grad=False) 
fs = torch.tensor(f, requires_grad=False) 
us = torch.tensor(u, requires_grad=False)

optimizer = torch.optim.SGD([u0, qc, qj], lr=1)

def fitness(u0, qj, qc):
    FR = Smulders(u0, qj, qc)
    u_pred = FR.u(q)
    if qc > qj or u0 < 0 or qj < 0 or qc < 0:
        return -np.linalg.norm(u_pred - u) * 10
    else:
        return -np.linalg.norm(u_pred - u)

history = []
u0s = []
qcs = []
qjs = []

epochs = 10
for _ in tqdm.tqdm(range(epochs)):
    f_pred = u0 / qj * (qc * qj - qc * qc + (qc + qs - qj) * relu(qc - qs) - qc * relu(qs - qc) )
    u_pred = f_pred / qs
    loss = mse_loss(f_pred, fs)
    # loss = mse_loss(u_pred, us)

    history.append(float(loss))
    # history.append(fitness(float(u0), float(qj), float(qc)))
    u0s.append(float(u0))
    qcs.append(float(qc))
    qjs.append(float(qj))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(u0, qc, qj)

    u0.grad.zero_()
    qc.grad.zero_()
    qj.grad.zero_()


print("Final Parameters:")
print("u0:", u0.item())
print("qc:", qc.item())
print("qj:", qj.item())
print("\nFinal u/q fitness:", fitness(float(u0), float(qj), float(qc)))

# plt.plot(qs.numpy(), fs.numpy(), ".")
# qtest = torch.linspace(0, float(qj), 100)
# ftest = u0 / qj * (qc * qj - qc * qc + (qc + qtest - qj) * relu(qc - qtest) - qc * relu(qtest - qc) )
# plt.plot(qtest.numpy(), ftest.detach().view_as(qtest).numpy())
# plt.show()

# plt.plot(u0s)
# plt.plot(qcs)
# plt.plot(qjs)
plt.plot(history)
# plt.ylim(-200, -140)
plt.yscale("log")
plt.show()
