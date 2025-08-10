# Imports
from resnet import ResNet1d
import tqdm
import h5py
import torch
import os
import json
import numpy as np
import argparse
from warnings import warn
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mdl', type=str,
                        help='folder containing model.', default='model')
    parser.add_argument('--path_to_traces', type=str, default='data/exams_part16_abs_age_1.hdf5',
                        help='path to hdf5 containing ECG traces.')
    parser.add_argument('--path_to_ages', type=str, default='data/age_part16_abs_age_1.npy',
                        help='path to npy containing ages of the respective patients.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='number of exams per batch.')
    parser.add_argument('--output', type=str, default='output/predicted_age.csv',
                        help='output file.')
    parser.add_argument('--traces_dset', default='tracings',
                         help='traces dataset in the hdf5 file.')
    parser.add_argument('--ids_dset',
                         help='ids dataset in the hdf5 file.')
    args, unk = parser.parse_known_args()
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Get checkpoint
    ckpt = torch.load(
        os.path.join(args.mdl, 'model.pth'),
        weights_only=False,
        map_location=lambda storage,
        loc: storage
    )
    # Get config
    config = os.path.join(args.mdl, 'config.json')
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

    # Get traces
    ff = h5py.File(args.path_to_traces, 'r')
    traces = ff[args.traces_dset]
    n_total = len(traces)
    if args.ids_dset:
        ids = ff[args.ids_dset]
    else:
        ids = range(n_total)

    # Read ages
    ages = np.load(args.path_to_ages)

    # Get dimension
    predicted_age = np.zeros((n_total,))
    reconstructed_input = np.zeros(traces.shape)

    # Evaluate on test data
    model.eval()
    n_total, n_samples, n_leads = traces.shape
    n_batches = int(np.ceil(n_total/args.batch_size))

    # Compute gradients
    predicted_age = np.zeros((n_total,))
    end = 0
    for i in tqdm.tqdm(range(n_batches)):
        start = end
        end = min((i + 1) * args.batch_size, n_total)
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
        print (f"x.shape: {x.shape}")
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
    df.to_csv(args.output)

    reconstructed_file = args.output.replace('.csv', '.npy')
    np.save(reconstructed_file, reconstructed_input)

