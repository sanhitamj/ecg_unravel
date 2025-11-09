import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm

from constants import (
    DATA_INPUT_DIR,
    N_LEADS,
)
from resnet import ResNet1d
from train import compute_loss, compute_weights

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


def predict(
        data_array,
        df,
        exam_ids,
        reconstruct=False,
        n_total=0,
        batch_size=10,
):
    if n_total == 0:
        n_total = len(data_array)
    n_batches = int(np.ceil(n_total/batch_size))

    pred_list = []
    predicted_age = np.zeros((n_total,))
    end = 0
    reconstructed_traces = []
    for i in tqdm.tqdm(range(n_batches)):
        start = end
        end = min((i + 1) * batch_size, n_total)

        # Get the predictions

        model.zero_grad()
        X = torch.tensor(
            data_array[start:end, :, :],
            requires_grad=reconstruct  # need this to retreive ECGs after backprop
        ).transpose(-1, -2)
        y_pred = model(X)
        if reconstruct:
            # X = torch.tensor(data_array[start:end, :, :], requires_grad=True).transpose(-1, -2)
            X.retain_grad()
            y_pred = model(X)

            ages = torch.from_numpy(df['age'][start:end].values)
            weights = torch.from_numpy(compute_weights(ages))
            loss = compute_loss(
                ages, y_pred, weights
            )
            loss.backward()
            reconstructed_traces.append(X.grad.detach().cpu().transpose(-1, -2).numpy().astype(np.float32))

        # Merge predictions back onto the metadata frame
        preds = pd.DataFrame({
            'exam_id': exam_ids[start:end],
            'torch_pred': y_pred.detach().numpy().squeeze()
        })

        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()
        pred_list.append(preds)

    preds = pd.concat(pred_list, axis=0, ignore_index=True)
    compare = df.merge(preds, on='exam_id', how='inner')
    print(compare[['age', 'nn_predicted_age', 'torch_pred']].head())
    if reconstruct:
        recon_traces = np.concatenate(reconstructed_traces, axis=0)
        print(f"recon_traces.shape: {recon_traces.shape}")
        np.save("reconstructed_traces.npy", recon_traces)
        plt.plot(recon_traces[0, :, 0], label='Reconstructed?')
        plt.plot(data_array[0, :, 0], label='original')
        plt.legend()
        plt.show()


    # Plot the new predictions against the metadata predictions

    plt.scatter(compare['nn_predicted_age'], compare['torch_pred'])
    plt.xlabel('NN Predicted Age')
    plt.ylabel('Torch Predicted Age')
    plt.savefig("plot.png")
    plt.show()


def main(n_total=0):
    # Read in exam metadata and limit to file 16.
    df = pd.read_csv(f'./{DATA_INPUT_DIR}/exams.csv')
    df = df[df['trace_file'] == 'exams_part16.hdf5']

    # Read in raw ECG data for file 16.
    filename = "./data/exams_part16.hdf5"

    with h5py.File(filename, "r") as f:
        print("Keys in the HDF5 file:", list(f.keys()))
        data_array = f['tracings'][()]
        exam_ids = f['exam_id'][()]

    # Brute force sort df to match exam_ids
    out_df = []
    for i in range(df.shape[0]):
        out_df.append(df.loc[df['exam_id'] == exam_ids[i]])
    df = pd.concat(out_df)

    if n_total == 0:
        df = df[
            (abs(df['nn_predicted_age'] - df['age']) < 1) &
            (df['normal_ecg'])
        ].copy()

        # Find indices of desired exam_ids
        mask = np.isin(exam_ids, df['exam_id'].values)
        exam_ids = exam_ids[mask]

        # Now read only the matching tracings
        data_array = data_array[mask, :, :]  # shape: (len(indices), ...)
        n_total = mask.sum()

    return data_array, df, exam_ids


if __name__ == "__main__":
    n_total = 0  # to use filters; use a positive number to use first n
    batch_size = 20
    data_array, df, exam_ids = main(n_total=n_total)
    predict(data_array, df, exam_ids, reconstruct=True, n_total=n_total, batch_size=batch_size)
