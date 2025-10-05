# How to read data from hdf5 files

import h5py
import json
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch

from constants import (
    DATA_INPUT_DIR,
    DATA_OUTPUT_DIR,
    AGE_FILTER,
    ABS_AGE_DIFF,
    EXAM_ID,
    FILE_NUM,
    N_LEADS,
    SAVE_HDF5,
)

from evaluate import predict
from resnet import ResNet1d

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


# setting up the model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mdl = 'model'

# Get checkpoint
try:
    ckpt = torch.load(
        os.path.join(mdl, 'model.pth'),
        weights_only=False,
        map_location=lambda storage,
        loc: storage
    )
except FileNotFoundError:
    # this is for Jupyter notebook
    mdl = '../model'
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

is_notebook = False

def get_exam_ids_per_file():
    """
    Returns file_path to read - the path will change according to file_num - hfd5 file
    and filtered dataframe according to filters
    """

    # Read the metadata file
    try:
        df = pd.read_csv(f"{DATA_INPUT_DIR}/exams.csv")
    except FileNotFoundError:
        # temporary fix for notebook
        is_notebook = True
        df = pd.read_csv(f"../{DATA_INPUT_DIR}/exams.csv")

    # Keep only the ones that have normal ECG and the absolute
    # difference between their chronological and predicted age < ABS_AGE_DIFF
    df = df[
        (abs(df['age'] - df['nn_predicted_age']) < ABS_AGE_DIFF) &
        (df['normal_ecg'])
    ].copy()

    # if AGE_FILTER, use only those that have that chronological age
    if AGE_FILTER:
        df = df[df['age'] == AGE_FILTER].copy()

    logging.info(f"Found {len(df)} patients in file num {FILE_NUM}.")
    trace_file_name = f"exams_part{FILE_NUM}.hdf5"
    if not is_notebook:
        trace_file_path = f"{DATA_INPUT_DIR}/{trace_file_name}"
    else:
        # current fix for using notebooks
        trace_file_path = f"../{DATA_INPUT_DIR}/{trace_file_name}"

    return trace_file_path, df[df['trace_file'] == trace_file_name]


def extract_selected_tracings(
        trace_file_path,
        ids_popln,
):

    logging.info(f"reading trace file: {trace_file_path}")

    try:
        with h5py.File(trace_file_path, 'r') as f:
            exam_ids = f[EXAM_ID][:]  # shape: (N,)
            tracings = f['tracings']  # don't load this yet; big dataset

            # Find indices of desired exam_ids
            mask = np.isin(exam_ids, ids_popln)
            indices = np.where(mask)[0]

            # Now read only the matching tracings
            selected_traces = tracings[indices]  # shape: (len(indices), ...)
            assert (len(selected_traces) == len(ids_popln)), \
                "The lengths of the arrays, indices and traces, do not match."

            p = Path(DATA_OUTPUT_DIR)
            # Create a directory if doesn't exist
            p.mkdir(parents=True, exist_ok=True)

            npy_path = f"{DATA_OUTPUT_DIR}/p{FILE_NUM}_age_diff_{ABS_AGE_DIFF}_orig.npy"
            if AGE_FILTER:
                npy_path = npy_path.replace("_orig.npy", f"_age_{AGE_FILTER}_orig.npy")
            # return selected_tracings, hdf5_path, npy_path

            # Don't save the npy file if the hfd5 file cannot be opened.
            if isinstance(selected_traces, np.ndarray):
                logging.info("Found tracings")
                with open(npy_path, 'wb') as f:
                    np.save(f, selected_traces)
                    logging.info("Saved npy file")
            else:
                logging.error("Did not find tracings; didn't save selected tracings npy file.")

            # If working with massive data, using hdf5 file.
            if SAVE_HDF5:
                hdf5_path = trace_file_path.split("/")[-1].replace(
                    ".hdf5", f"_abs_age_{ABS_AGE_DIFF}.hdf5"
                )
                hdf5_path = f"{DATA_OUTPUT_DIR}/{hdf5_path}"
                with h5py.File(hdf5_path, 'w') as f:
                    f.create_dataset('exam_id', data=ids_popln)
                    f.create_dataset('tracings', data=selected_traces, compression='gzip')
        return selected_traces

    except OSError:
        logging.error(f"File {trace_file_path} corrupted. Download again.")


def reconstruct_traces(
        traces,
        df_metadata,
        model,
        recon_ages=[age for age in range(20, 81)]
):
    if traces.ndim == 3:
        assert (len(traces) == len(df_metadata)), \
            "Lengths of traces and metadata do not match."
    if traces.ndim == 2:
        assert(len(df_metadata) == 1), \
            "Too many patients in the metadata dataframe"
    ids_popln = df_metadata[EXAM_ID].values
    real_ages = df_metadata['age'].values

    # recon_ages = [age for age in range(20, 81)]

    # If passing multiple patients save numpy files, one file per patient
    if traces.ndim == 3:
        for i, exam_id in enumerate(ids_popln):
            recon_trace = predict(
                model,
                traces[i, :, :],
                recon_ages,
            )
            recon_file_path = f"{DATA_OUTPUT_DIR}/id_{exam_id}_age_{real_ages[i]}_recon.npy"
            np.save(recon_file_path, recon_trace)

    # If passing one patient, return numpy array of reconstructed traces
    if traces.ndim == 2:
        recon_trace = predict(
            model,
            traces,
            recon_ages,
        )
        # recon_trace should be only 2-dim array if only one recon_age is passed
        if len(recon_ages) == 1:
            recon_trace = recon_trace[0, :, :]
        return recon_trace


if __name__ == "__main__":
    trace_file_path, df = get_exam_ids_per_file()
    df.to_csv(f"{DATA_OUTPUT_DIR}/exams_metadata.csv", index=False)
    selected_traces = extract_selected_tracings(
        trace_file_path,
        df[EXAM_ID]
    )
    reconstruct_traces(selected_traces, df, model)
