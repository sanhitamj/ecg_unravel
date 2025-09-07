import logging
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def save_model_weights_csv(
        pth_file='model.pth',
        preserve_shape=True,
        csv_file="model_weights.csv"
):
    """
    To save model weights in CSV.

    If running from the notebook directory, use pth_file='../model.pth'
    preserve_shape=True will save one row per layer of the NN.
    False will save one row per neuron

    If csv_file is passed, it'll save the save the dataframe.
    If not, it will return the dataframe.

    """

    # Load the PyTorch model weights
    state_dict = torch.load(pth_file, map_location=torch.device(device=device))

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    rows = []

    if preserve_shape:
        for name, param in state_dict.items():
            tensor = param.cpu().numpy()
            shape = list(tensor.shape)
            flattened = tensor.flatten().tolist()
            rows.append({
                "parameter": name,
                "shape": shape,
                "values": flattened
            })
        columns = ["parameter", "shape", "values"]

    else:
        for name, param in state_dict.items():
            # Convert tensor to numpy
            param_np = param.cpu().numpy().flatten()
            for i, val in enumerate(param_np):
                rows.append([name, i, val])
        columns = ["parameter", "index", "values"]

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=columns)

    if csv_file:
        df.to_csv(csv_file, index=False)
        logging.info(f"Model weights saved to {csv_file}")
    else:
        return df
