# How to read data from hdf5 files

import h5py
import numpy as np
import pandas as pd
from tempfile import TemporaryFile

DATA_DIR = "data"
ABSOLUTE_AGE_DIFF = 1
EXAM_ID = 'exam_id'


def get_exam_ids_per_file(file_num=16):
    # Read the metadata file
    df = pd.read_csv(f"{DATA_DIR}/exams.csv")

    # Keep only the ones that have normal ECG and the absolute
    # difference between their chronological and predicted age < 1
    df = df[
        (abs (df['age']- df['nn_predicted_age']) < ABSOLUTE_AGE_DIFF) &
        (df['normal_ecg'])
    ].copy()

    roi_files = df['trace_file'].value_counts()

    trace_file_name = f"exams_part{file_num}.hdf5"
    trace_file_path = f"{DATA_DIR}/{trace_file_name}"
    return trace_file_path, df[df['trace_file'] == trace_file_name][EXAM_ID]


def extract_selected_tracings(trace_file_path, ids_pop):

    with h5py.File(trace_file_path, 'r') as f:
        exam_ids = f[EXAM_ID][:]  # shape: (N,)
        tracings = f['tracings']  # don't load this yet; big dataset

        # Find indices of desired exam_ids
        mask = np.isin(exam_ids, ids_pop)
        indices = np.where(mask)[0]

        # Now read only the matching tracings
        selected_tracings = tracings[indices]  # shape: (len(indices), ...)
        print(f"selected_tracings.shape {selected_tracings.shape}")
        return selected_tracings


if __name__ == "__main__":
    file_num = 16
    trace_file_path, ids_pop = get_exam_ids_per_file(file_num=file_num)
    selected_tracings = extract_selected_tracings(trace_file_path, ids_pop)
    outfile_path = trace_file_path.replace(".hfd5", "_sel.npy")
    
    with open(outfile_path, 'w') as f:
        np.save(f, selected_tracings)
    