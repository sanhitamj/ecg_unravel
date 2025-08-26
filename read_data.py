# How to read data from hdf5 files

import h5py
import logging
import numpy as np
import pandas as pd
from pathlib import Path


from constants import (
    FILE_NUM,
    DATA_INPUT_DIR,
    DATA_OUTPUT_DIR,
    AGE_FILTER,
    ABS_AGE_DIFF,
    EXAM_ID,
)


def get_exam_ids_per_file(
        file_num
):
    """
    Takes in the the file number to process
    Returns file_path to read and exam_ids to read
    """

    # Read the metadata file
    df = pd.read_csv(f"{DATA_INPUT_DIR}/exams.csv")

    # Keep only the ones that have normal ECG and the absolute
    # difference between their chronological and predicted age < ABS_AGE_DIFF
    df = df[
        (abs(df['age'] - df['nn_predicted_age']) < ABS_AGE_DIFF) &
        (df['normal_ecg'])
    ].copy()

    # if AGE_FILTER, use only those that have that chronological age
    if AGE_FILTER:
        df = df[df['age'] == AGE_FILTER].copy()

    trace_file_name = f"exams_part{file_num}.hdf5"
    trace_file_path = f"{DATA_INPUT_DIR}/{trace_file_name}"
    return trace_file_path, df[df['trace_file'] == trace_file_name][EXAM_ID]


def extract_selected_tracings(trace_file_path, ids_popln):

    logging.info(f"reading trace file: {trace_file_path}")

    try:
        with h5py.File(trace_file_path, 'r') as f:
            exam_ids = f[EXAM_ID][:]  # shape: (N,)
            tracings = f['tracings']  # don't load this yet; big dataset

            # Find indices of desired exam_ids
            mask = np.isin(exam_ids, ids_popln)
            indices = np.where(mask)[0]

            # Now read only the matching tracings
            selected_tracings = tracings[indices]  # shape: (len(indices), ...)
            logging.info(f"selected_tracings.shape: {selected_tracings.shape}")
            logging.info(f"len(ids_popln): {len(ids_popln)}")
            assert (len(selected_tracings) == len(ids_popln)), \
                "The lengths of the arrays, indices and traces, do not match."

            p = Path(DATA_OUTPUT_DIR)
            p.mkdir(parents=True, exist_ok=True)
            hdf5_path = trace_file_path.split("/")[-1].replace(
                ".hdf5", f"_abs_age_{ABS_AGE_DIFF}.hdf5"
            )
            hdf5_path = f"{DATA_OUTPUT_DIR}/{hdf5_path}"

            npy_path = hdf5_path.replace("hdf5", "npy")
            # return selected_tracings, hdf5_path, npy_path

            # Don't try to save the npy file if the hfd5 file cannot be opened.
            if isinstance(selected_tracings, np.ndarray):
                logging.info("Found tracings")
                with open(npy_path, 'wb') as f:
                    np.save(f, selected_tracings)
                    logging.info("Saved npy file")
            else:
                logging.error("Did not find tracings; didn't save selected tracings npy file.")

            with h5py.File(hdf5_path, 'w') as f:
                f.create_dataset('exam_id', data=ids_popln)
                f.create_dataset('tracings', data=selected_tracings, compression='gzip')

    except OSError:
        logging.error(f"File {trace_file_path} corrupted. Download again.")


def write_selected_input_files():
    trace_file_path, ids_popln = get_exam_ids_per_file(file_num=FILE_NUM)
    extract_selected_tracings(trace_file_path, ids_popln)

    # Write numpy files for Deejay


if __name__ == "__main__":
    write_selected_input_files()
