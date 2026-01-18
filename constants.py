# Save the files downloaded from zenodo in `DATA_INPUT_DIR`
DATA_DIR = './data'
MODEL_DIR = './model'

N_LEADS = 12  # 12 channels for ECG
# N_TOTAL = 0  # make this 0 to use filters
# RECONSTRUCT = False  # To create a file with the reconstructed traces

OUTPUT_PREDICTIONS = f'{DATA_DIR}/predictions.csv'
RECONSTRUCT_FILE = f'{DATA_DIR}/reconstructed_traces.npy'

# using 16 as default - that has the most data with the following 2 constraints:
# ABS_AGE_DIFF and NORMAL_ECG

ABS_AGE_DIFF = 1  # the absolute difference between chronological and predicted ages
# NORMAL_ECG = True

# AGE_FILTER = False  # using patients with this chronological age.
