import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from constants import (
    # ABS_AGE_DIFF,
    DATA_DIR,
    MODEL_DIR,
    N_LEADS,
)
from resnet import ResNet1d
from train import compute_loss, compute_weights

config = f'{MODEL_DIR}/config.json'

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
state_dict = (torch.load(f'{MODEL_DIR}/model.pth',
              weights_only=False,
              map_location=device))

# Load the state dict and set the model to eval mode.
model.load_state_dict(state_dict['model'])
model.eval()


def predict(
        data_array,
        df=pd.DataFrame,
        exam_ids=np.array([]),
        reconstruct=False,
        batch_size=8,
        reconstruct_file='reconstruct_16.npy',
):
    """
    data_array: 3-D array with patients and their 2-D ECGs
    df: optional, metadata or part of it from exams.csv
    exam_ids: optional, exam_ids for the patients in data_array
    reconstruct: if True, reconstructed ECGs from the neural net are written
    batch_size: for prediction


    If not given df and exam_ids, returns a prediction array
    """
    data_array = data_array.astype(np.float32)
    n_total = len(data_array)
    n_batches = int(np.ceil(n_total/batch_size))

    pred_list = []
    predicted_age = np.zeros((n_total,))
    end = 0
    reconstructed_traces = []
    for i in range(n_batches):
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
            X.retain_grad()
            y_pred = model(X)

            ages = torch.from_numpy(df['age'][start:end].values)
            weights = torch.from_numpy(compute_weights(ages))
            loss = compute_loss(
                ages, y_pred, weights
            )
            loss.backward()
            reconstructed_traces.append(
                X.grad.detach().cpu().transpose(-1, -2).numpy().astype(np.float32)
            )

        # Merge predictions back onto the metadata frame
        if exam_ids.any():
            preds = pd.DataFrame({
                'exam_id': exam_ids[start:end],
                'torch_pred': y_pred.detach().numpy().squeeze()
            })
            pred_list.append(preds)

        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()

    if exam_ids.any():
        preds = pd.concat(pred_list, axis=0, ignore_index=True)
        preds = df.merge(preds, on='exam_id', how='inner')

        preds.to_csv(f"{DATA_DIR}/prediction.csv", index=False)
    else:
        return predicted_age

    if reconstruct:
        recon_traces = np.concatenate(reconstructed_traces, axis=0)
        print(f"recon_traces.shape: {recon_traces.shape}")
        np.save(reconstruct_file, recon_traces)
        plt.plot(recon_traces[0, :, 0], label='Reconstructed')
        plt.plot(data_array[0, :, 0], label='original')
        plt.legend()
        plt.show()

    # Plot the new predictions against the metadata predictions

    plt.scatter(preds['nn_predicted_age'], preds['torch_pred'])
    plt.xlabel('NN Predicted Age')
    plt.ylabel('Torch Predicted Age')
    plt.savefig("plot.png")
    plt.show()
    return


def read_data(
        n_total=1000,
        data_file='trace_file.npy',
        abs_age_diff=100,
    ):
    """
    Takes in 3 arguments:
    n_total: either 0 to use filters, or a positive integer
    data_file: save those selected traces in this file
    abs_age_diff: to be used in filters; default is 100 to allow all the patients
    """

    # Read in exam metadata and limit to file 16.
    df = pd.read_csv(f'{DATA_DIR}/exams.csv')
    df = df[df['trace_file'] == 'exams_part16.hdf5']

    # Read in raw ECG data for file 16.
    filename = f"{DATA_DIR}/exams_part16.hdf5"

    with h5py.File(filename, "r") as f:
        print("Keys in the HDF5 file:", list(f.keys()))
        data_array = f['tracings'][()]
        exam_ids = f['exam_id'][()]

    # Sort df to match exam_ids
    df = df.iloc[[list(exam_ids).index(x)
                  if x in list(exam_ids) else None
                  for x in df['exam_id']]]

    if n_total == 0:
        df = df[
            (abs(df['nn_predicted_age'] - df['age']) < abs_age_diff) &
            (df['normal_ecg'])
        ].copy()

        # Find indices of desired exam_ids
        mask = np.isin(exam_ids, df['exam_id'].values)
        exam_ids = exam_ids[mask]

        # Now read only the matching tracings
        data_array = data_array[mask, :, :]  # shape: (len(indices), ...)
        n_total = mask.sum()

    else:
        data_array = data_array[:n_total]

    if data_file:
        np.save(data_file, data_array)
    return data_array, df, exam_ids


def predict_with_removal(
        data_array,
        start,
        interval,
        replace_near=True
    ):
    """
    Docstring for pred_with_remove

    :param data_array: 3-d numpy array with traces with single averaged beat
    :param idx: at what index to start removal of the data
    :param interval: how many points to remove
    """
    end = start + interval  # remove the original values for this range of pixels
    replace_idx = start - 1  # replace those with the value from this pixel

    for i in range(len(data_array)):
        for chan in range(12):
            if replace_near:
                replace_val = data_array[i, replace_idx, chan]
            else:
                replace_val = np.mean(data_array[i, start, chan], data_array[i, end, chan])
            data_array[i, start:end, chan] = replace_val
    return predict(data_array)


def calculate_removal_error(
        data_array_loc,
        interval,
        n=1000,
    ):
    """
    Docstring for removal_error

    :param data_array: 3-d array of traces with a single averaged beat
    :param interval: how many pixels to remove
    :param n: use first n values of the array for predictions; if 0 means use the whole array
    """

    data_array = np.load(data_array_loc)
    if n > 0:
        data_array = data_array[:n, :, :]
    avg_pred = predict(data_array)
    rmses = []
    start_pixels = []
    counter = 0
    for start in range(1775, 2191, 10):
        # Using these ends as start_max and end_min for all the subjects, in the averaged beat
        data_array = np.load(data_array_loc)
        if n > 0:
            data_array = data_array[:n, :, :]
        out = predict_with_removal(data_array, start, interval=interval)
        start_pixels.append(start)
        rmses.append(float(np.sqrt(np.sum(avg_pred - out) ** 2)))
        counter += 1
        if counter % 50 == 0:
            print(f"Iteration {counter} for start pixel {start} done.")
    return pd.DataFrame({
        'start_pixel': start_pixels,
        'rmse': rmses
    })


if __name__ == "__main__":
    batch_size = 20
    data_array, df, exam_ids = read_data(n_total=100, data_file='trace_file.npy')
    predict(data_array, df, exam_ids, reconstruct=False, batch_size=batch_size)
