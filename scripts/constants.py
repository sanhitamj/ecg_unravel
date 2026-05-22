# Save the files downloaded from zenodo in `DATA_INPUT_DIR`
DATA_DIR = '../data'
MODEL_DIR = '../model'
OUTPUT_DIR = '../output'

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

# PQRST sections of the averaged hearbeat trace, for plotting only
waves = {
    'P': [(1970, 2020), 'green', 'solid'],
    'Q': [(2030, 2035), 'red', 'solid'],
    'R-asc-pre': [(2036, 2042), 'blue', 'solid'],
    'R-asc-post': [(2043, 2048), 'darkred', 'dashed'],
    'R-desc-pre': [(2049, 2054), 'darkcyan', 'dotted'],
    'R-desc-post': [(2055, 2063), 'aqua', 'dashdot'],
    'ST': [(2064, 2119), 'purple', 'solid'],
    'T': [(2120, 2180), 'orange', 'solid'],
    'P-Q': [(2020, 2030), 'olive', 'solid']
}
