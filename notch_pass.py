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
    DATA_DIR,
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
df = pd.read_csv(f'./data/exams.csv')
df = df[df['trace_file'] == 'exams_part16.hdf5']

# Read in raw ECG data for file 16.
filename = f"./data/exams_part16.hdf5"

with h5py.File(filename, "r") as f:
    print("Keys in the HDF5 file:", list(f.keys()))
    dataset = f['tracings']
    print("Dataset shape:", dataset.shape)
    print("Dataset dtype:", dataset.dtype)
    data_array = f['tracings'][()]
    exam_ids = f['exam_id'][()]

samp_freq = 409.6  # Sample frequency (Hz)
quality_factor = 20.0  # Quality factor

n_total = 1000  # total number of predictions
batch_size = 10
n_batches = int(np.ceil(n_total/batch_size))

data_array_trans = np.zeros_like(data_array[:n_total, :, :])

mse = []
for freq in range(2, 50):
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
                            'torch_pred': y_pred.detach().numpy().squeeze()})
        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
        pred_list.append(preds)

    preds = pd.concat(pred_list, axis=0, ignore_index=True)
    compare = df.merge(preds, on='exam_id', how='inner')
    mse.append(float(np.mean((compare['nn_predicted_age'] - compare['torch_pred'])**2)))

os.makedirs('./output/images', exist_ok=True)
plt.plot(np.arange(2, len(mse) + 2), mse)
plt.xlabel('Frequency')
plt.ylabel('MSE')
plt.savefig('./output/images/frequency_vs_mse.png')
plt.show()
