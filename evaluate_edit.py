import json
import logging
import numpy as np
import os
import pandas as pd
import torch
import tqdm

from constants import (
    batch_size,
    KEEP_AGE,
    N_LEADS,
)
from resnet import ResNet1d

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def predict(
        model,   # original model object
        traces,  # original traces
        ages,  # ages to reconstruct
):
    # If using an age range for reconstructing traces
    if not KEEP_AGE:
        assert (traces.ndim == 2), "Use one patient trace if using age-range to reconstruct traces."
        if traces.ndim == 2:
            traces = np.repeat(traces[None, :, :], len(ages), axis=0)
            # repeat the traces array to run eval in batches
        ids = [0]
    else:
        ids = range(len(traces))

    n_total = len(traces)

    # Get dimension
    predicted_age = np.zeros((n_total,))
    recon_trace = np.zeros(traces.shape)

    # Evaluate on test data
    model.eval()
    # n_total, n_samples, n_leads = traces.shape
    n_batches = int(np.ceil(n_total/batch_size))

    # Compute gradients
    end = 0
    for i in tqdm.tqdm(range(n_batches)):
        start = end
        end = min((i + 1) * batch_size, n_total)
        # with torch.no_grad():

        x_leaf = torch.tensor(traces[start:end, :, :], dtype=torch.float32, requires_grad=True, device=device)

        # Transpose for model input (now non-leaf, so we must retain its grad)
        x = x_leaf.transpose(-1, -2)
        x.retain_grad()  # Needed to get grad for non-leaf tensor

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
        recon_trace[start:end] = x.grad.detach().cpu().transpose(-1, -2).numpy().astype(np.float32)

        # Zero gradients before next batch
        model.zero_grad()

        # print(f"reconstructed_input.shape = {input_gradients}")
    # Save predictions

    if KEEP_AGE:
        df = pd.DataFrame({'ids': ids, 'predicted_age': predicted_age})
        df = df.set_index('ids')
        logging.info(f"Output csv shape: {df.shape}")
        df.to_csv("test_prdicted_age.csv", index=False)

    return recon_trace


if __name__ == "__main__":
    # Just to check if this file works

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mdl = 'model'

    # Get checkpoint
    ckpt = torch.load(
        os.path.join(mdl, 'model.pth'),
        weights_only=False,
        map_location=lambda storage,
        loc: storage
    )
    config = os.path.join(mdl, 'config.json')
    with open(config, 'r') as f:
        config_dict = json.load(f)
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

    traces = np.load("output/part16_age_diff_1_age_20/exams_part16_abs_age_1.npy")

    recon_trace = predict(
        model,
        traces[0, :, :],
        ages=[x for x in range(20, 81)]
    )
    np.save("test_recon_age.npy", recon_trace)
    logging.info("Saved a test file: test_recon_age.npy")
