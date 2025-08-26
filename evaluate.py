import h5py
import torch
import os
import json
import logging
import numpy as np
import pandas as pd
import tqdm

from constants import (
    ABS_AGE_DIFF,
    batch_size,
    # DATA_INPUT_DIR,
    DATA_OUTPUT_DIR,
    FILE_NUM,
    PREDICTED_AGE_CSV,
    REPLACE_AGE,
    REPLACE_AGE_RANGE,
    RECONSTED_ECG,
    traces_dset
)
from resnet import ResNet1d

# default_input_file_   name = f'{DATA_INPUT_DIR}/exams_part{FILE_NUM}_abs_age_{ABS_AGE_DIFF}.hdf5'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mdl = 'model'

# Get checkpoint
ckpt = torch.load(
    os.path.join(mdl, 'model.pth'),
    weights_only=False,
    map_location=lambda storage,
    loc: storage
)


def predict():
    # # Get config

    config = os.path.join(mdl, 'config.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
    # Get model
    N_LEADS = 12
    model = ResNet1d(
        input_dim=(N_LEADS, config_dict['seq_length']),
        blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
        n_classes=1,
        kernel_size=config_dict['kernel_size'],
        dropout_rate=config_dict['dropout_rate']
    )

    # load model checkpoint
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    path_to_traces = f'{DATA_OUTPUT_DIR}/exams_part{FILE_NUM}_abs_age{ABS_AGE_DIFF}_abs_age{ABS_AGE_DIFF}.hdf5'
    path_to_traces = "output/part16_age_diff_1_age_20/exams_part16_abs_age_1.hdf5"

    # Get traces
    ff = h5py.File(path_to_traces, 'r')
    traces = ff[traces_dset]
    # traces = np.load(f'{DATA_OUTPUT_DIR}/exams_part{FILE_NUM}_abs_age{ABS_AGE_DIFF}.npy')
    logging.info(f"type(trace): {type(traces)}")
    n_total = len(traces)
    ids = range(n_total)

    KEEP_AGE = False
    path_to_ages = False
    if KEEP_AGE:
        ages = np.load(path_to_ages)
    elif REPLACE_AGE:
        ages = np.array([REPLACE_AGE] * n_total)
    else:
        # If the age range is used use only one ID at a time, for now.
        ages = np.array([age for age in REPLACE_AGE_RANGE])
        traces = np.repeat(traces[None, :], n_total, axis=0)
        ids = [0]

    # Read ages
    # ages = np.load(args.path_to_ages)

    # Get dimension
    predicted_age = np.zeros((n_total,))
    reconstructed_input = np.zeros(traces.shape)

    # Evaluate on test data
    model.eval()
    n_total, n_samples, n_leads = traces.shape
    n_batches = int(np.ceil(n_total/batch_size))

    # Compute gradients
    predicted_age = np.zeros((n_total,))
    end = 0
    for i in tqdm.tqdm(range(n_batches)):
        start = end
        end = min((i + 1) * batch_size, n_total)
        # with torch.no_grad():

        x_leaf = torch.tensor(traces[start:end, :, :], dtype=torch.float32, requires_grad=True, device=device)

        # Transpose for model input (now non-leaf, so we must retain its grad)
        x = x_leaf.transpose(-1, -2)
        x.retain_grad()  # Needed to get grad for non-leaf tensor

        # x = torch.tensor(
        #     traces[start:end, :, :],
        #     requires_grad=True
        # ).transpose(-1, -2)
        # x = x.to(device, dtype=torch.float32)
        print(f"x.shape: {x.shape}")
        y_pred = model(x)

        # Example target tensor for backprop — must match your task
        y_true = torch.tensor(
            ages[start:end],
            device=device,
            dtype=torch.float32
        ).unsqueeze(1)

        # Loss and backward pass
        criterion = torch.nn.MSELoss()
        loss = criterion(y_pred, y_true)
        loss.backward()

        # Store predictions
        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten()

        # Store backpropagated gradients (input reconstruction)
        reconstructed_input[start:end] = x.grad.detach().cpu().transpose(-1, -2).numpy()

        # Zero gradients before next batch
        model.zero_grad()

        # print(f"reconstructed_input.shape = {input_gradients}")
    # Save predictions
    df = pd.DataFrame({'ids': ids, 'predicted_age': predicted_age})
    df = df.set_index('ids')
    print(f"Output csv shape: {df.shape}")
    df.to_csv(PREDICTED_AGE_CSV, index=False)
    logging.info(f"reconstructed_input shape: {reconstructed_input.shape}")

    np.save(RECONSTED_ECG, reconstructed_input)


if __name__ == "__main__":
    predict()
