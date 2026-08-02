import h5py
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from resnet import ResNet1d
from scipy import signal
import tqdm
from warnings import warn

from constants import (
    N_LEADS,
)
config = './model/config.json'

# Define the sample frequency. It's 4096 observations over 10 seconds
samp_freq = 409.6  # Sample frequency (Hz)

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

# Read in exam metadata.
df = pd.read_csv(f'./data/exams.csv')

quality_factor = 2.0  # Quality factor

batch_size = 10
all_compares = []

for i in range(18):

    filename = f"./data/exams_part{i}.hdf5"

    with h5py.File(filename, "r") as f:
        print(f"Raw file: {i}")
        print("Keys in the HDF5 file:", list(f.keys()))
        data_array = f['tracings'][()]
        exam_ids = f['exam_id'][()]
        n_total = data_array.shape[0]

    data_array_trans = np.zeros_like(data_array[:n_total, :, :])

    mse = []
    n_batches = int(np.ceil(n_total/batch_size))
    for freq in range(2, 51):
        notch_freq = freq  # Frequency to be removed from signal (Hz)
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
        for i in range(n_total):
            for j in range(data_array.shape[2]):
                data_array_trans[i, :, j] = signal.filtfilt(b_notch, a_notch, data_array[i, :, j])
        
        pred_list = []
        predicted_age = np.zeros((n_total,))
        end = 0
        for i in tqdm.tqdm(range(n_batches)):
            start = end
            end = min((i + 1) * batch_size, n_total)

            # Get the predictions

            model.zero_grad()
            y_pred = model(torch.tensor(data_array_trans[start:end, :, :]).transpose(-1, -2))

            # Merge predictions back onto the metadata frame
            preds = pd.DataFrame({'exam_id': exam_ids[start:end],
                                  f'torch_pred_freq_{freq}': y_pred.detach().numpy().squeeze()})
            predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
            pred_list.append(preds)

        preds = pd.concat(pred_list, axis=0, ignore_index=True)
        if freq == 2:
            compare = df.merge(preds, on='exam_id', how='inner')
        else:
            compare = compare.merge(preds, on='exam_id', how='inner')

    all_compares.append(compare)
    del data_array
    del exam_ids
    del data_array_trans

compare = pd.concat(all_compares, axis=0, ignore_index=True)

os.makedirs('./output', exist_ok=True)
compare.to_csv('./output/compares.csv', index=False)

mse = []
for freq in range(2, 50):
    mse.append(float(np.mean((compare['nn_predicted_age'] - compare[f'torch_pred_freq_{freq}'])**2)))

os.makedirs('./output/images', exist_ok=True)
plt.plot(np.arange(2, len(mse) + 2), mse)
plt.xlabel('Frequency')
plt.ylabel('MSE')
plt.savefig('./output/images/frequency_vs_mse.png')
plt.show()
