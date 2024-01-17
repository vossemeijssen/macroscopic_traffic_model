import godunovfunctions
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import csv
import time
from tqdm import tqdm


# Loading all data
data_name = "a13_2_months"
# datafolder = os.path.join(os.getcwd(), "data", data_name)
# df = godunovfunctions.load_data(datafolder, print_logs=True)
# torch.save(df, "gunodov_method/df_a13_2_months.pth")
df = torch.load("gunodov_method/df_"+data_name+".pth")
df = df.dropna(subset=['gem_dichtheid'])
X_df = df[[1, 2, 3, 4, 5, 6, "gem_dichtheid"]]
Y_df = df["gem_intensiteit"]


def train_and_save(
    model_linear_stack = nn.Sequential(
            nn.Linear(6, 4),
            nn.Softplus(),
            nn.Linear(4, 3),
            nn.Softplus(),
        ),
    X_min_normalizer = 0.0,
    X_max_normalizer = 100.0,
    Y_normalizer = 10000.0,
    criterion_function = nn.MSELoss,
    optimizer_function = optim.Adam,
    bias_init_function = nn.init.zeros_,
    weights_init_function = nn.init.xavier_uniform_,
    lr = 0.01,
    epochs = 1000,
    batch_size = 1000,
    k_folds = 5,):
    # ----------- From here, the hyperparameter search loop starts -------------
    # Normalize data, create tensors and create dataloader
    X = (torch.tensor(X_df.values, dtype=torch.float32) - X_min_normalizer) / (X_max_normalizer - X_min_normalizer)
    Y = torch.tensor(Y_df.values, dtype=torch.float32).view(-1) / Y_normalizer
    dataset = TensorDataset(X, Y)

    # We will save all results in logging_data
    logging_data = []

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=batch_size, sampler=test_subsampler)
        
        # Set up model and optimizer
        torch.manual_seed(42)
        model = godunovfunctions.NeuralNetwork(
            lin_stack=model_linear_stack,
            bias_init_function=bias_init_function,
            weights_init_function=weights_init_function)
        optimizer = optimizer_function(model.parameters(), lr=lr)
        criterion = criterion_function()

        # Run epochs
        starttime = time.time()
        for epoch in tqdm(range(epochs), desc=f"Epochs in fold {fold}", ncols=100):
            epoch_starttime = time.time()

            # Loop over all training batches
            train_loss_total = 0
            tot_batches = 0
            for x_batch, y_batch in trainloader:
                # Get train loss
                y_batch_pred  = model(x_batch)
                loss = criterion(y_batch_pred , y_batch)

                # Save that loss
                train_loss_total += loss.item()
                tot_batches += 1

                # Take a step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Scale loss according to batches
            train_loss_total = train_loss_total / tot_batches

            # Loop over all test batches
            test_loss_total = 0
            tot_batches = 0
            for x_batch, y_batch in testloader:
                # Get test loss
                y_batch_pred = model(x_batch)
                loss = criterion(y_batch_pred, y_batch)

                # Save that loss
                test_loss_total += loss.item()
                tot_batches += 1
            
            # Scale loss according to batches
            test_loss_total = test_loss_total / tot_batches

            # Save the epoch results
            logging_data.append([fold, epoch, train_loss_total, test_loss_total, time.time() - epoch_starttime, time.time() - starttime])

    # Save the logging_data in a csv
    setting_data = [
        data_name,
        str(model_linear_stack).replace("\n", ""),
        X_min_normalizer,
        X_max_normalizer,
        Y_normalizer,
        str(criterion_function),
        str(optimizer_function),
        str(bias_init_function),
        str(weights_init_function),
        lr,
        epochs,
        batch_size,
        k_folds
    ]

    with open("hyperparameter_logs.csv", "a", newline="") as f_object:
        writer_object = csv.writer(f_object)
        for row in logging_data:
            writer_object.writerow(setting_data + row)
        f_object.close()


lr = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
stacks = [
    nn.Sequential(
            nn.Linear(6, 3),
            nn.Softplus(),
    ),
    nn.Sequential(
            nn.Linear(6, 4),
            nn.Softplus(),
            nn.Linear(4, 3),
            nn.Softplus(),
    ),
    nn.Sequential(
            nn.Linear(6, 10),
            nn.Softplus(),
            nn.Linear(10, 4),
            nn.Softplus(),
            nn.Linear(4, 3),
            nn.Softplus(),
    ),
    nn.Sequential(
            nn.Linear(6, 20),
            nn.Softplus(),
            nn.Linear(20, 10),
            nn.Softplus(),
            nn.Linear(10, 4),
            nn.Softplus(),
            nn.Linear(4, 3),
            nn.Softplus(),
    ),
    nn.Sequential(
            nn.Linear(6, 40),
            nn.Softplus(),
            nn.Linear(40, 20),
            nn.Softplus(),
            nn.Linear(20, 10),
            nn.Softplus(),
            nn.Linear(10, 4),
            nn.Softplus(),
            nn.Linear(4, 3),
            nn.Softplus(),
    ),
]

setupcounter = 1
for stack in stacks:
    print(f"\nTraining and testing setup {setupcounter}")
    train_and_save(model_linear_stack=stack, epochs=10)
    setupcounter += 1 








# # Settings
# model_linear_stack = nn.Sequential(
#             nn.Linear(6, 50),
#             nn.Softplus(),
#             nn.Linear(50, 25),
#             nn.Softplus(),
#             nn.Linear(25, 10),
#             nn.Softplus(),
#             nn.Linear(10, 3),
#             nn.Softplus(),
#         )
# X_min_normalizer = 0.0
# X_max_normalizer = 100.0
# Y_normalizer = 10000.0
# criterion_function = nn.MSELoss
# optimizer_function = optim.Adam
# bias_init_function = nn.init.zeros_
# weights_init_function = nn.init.xavier_uniform_
# lr = 0.01
# epochs = 20
# batch_size = 1000
# k_folds = 5

