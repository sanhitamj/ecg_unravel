from resnet import ResNet1d
import tqdm
import h5py
import torch
import os
import json
import numpy as np
from warnings import warn
import pandas as pd

import matplotlib.pyplot as plt

from constants import (
    DATA_INPUT_DIR,
    N_LEADS,
)
config = './model/config.json'

# Instantiate the model using the config.json information.
with open(config, 'r') as f:
    config_dict = json.load(f)
model = ResNet1d(
    input_dim=(N_LEADS, config_dict['seq_length']),
    blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
    n_classes=1,
    kernel_size=config_dict['kernel_size'],
    dropout_rate=config_dict['dropout_rate']
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Retrieve the state dict, which has all the coefficients
state_dict = (torch.load('./model/model.pth',
              weights_only=False,
              map_location=device))

# Load the state dict and set the model to eval mode.
model.load_state_dict(state_dict['model'])
model.eval()

# Read in exam metadata and limit to file 16.
df = pd.read_csv(f'./{DATA_INPUT_DIR}/exams.csv')
df = df[df['trace_file'] == 'exams_part16.hdf5']

# Read in raw ECG data for file 16.
filename = "./data/exams_part16.hdf5"

with h5py.File(filename, "r") as f:
    print("Keys in the HDF5 file:", list(f.keys()))
    dataset = f['tracings']
    print("Dataset shape:", dataset.shape)
    print("Dataset dtype:", dataset.dtype)
    data_array = f['tracings'][()]
    exam_ids = f['exam_id'][()]


n_total = 1000  # total number of predictions
batch_size = 10
n_batches = int(np.ceil(n_total/batch_size))

pred_list = []
predicted_age = np.zeros((n_total,))
end = 0
for i in tqdm.tqdm(range(n_batches)):
    start = end
    end = min((i + 1) * batch_size, n_total)

    # Get the predictions

    model.zero_grad()
    y_pred = model(torch.tensor(data_array[start:end, :, :]).transpose(-1, -2))

    # Merge predictions back onto the metadata frame
    preds = pd.DataFrame({'exam_id': exam_ids[start:end],
                        'torch_pred': y_pred.detach().numpy().squeeze()})

    if i == 0:
        print(y_pred.detach().numpy().squeeze())

    predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
    pred_list.append(preds)


preds = pd.concat(pred_list, axis=0, ignore_index=True)
compare = df.merge(preds, on='exam_id', how='inner')


# Plot the new predictions against the metadata predictions
plt.scatter(compare['nn_predicted_age'], compare['torch_pred'])
plt.xlabel('NN Predicted Age')
plt.ylabel('Torch Predicted Age')
plt.savefig("plot.png")
plt.show()
