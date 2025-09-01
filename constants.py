EXAM_ID = 'exam_id'
DATA_INPUT_DIR = 'data'

FILE_NUM = 16
ABS_AGE_DIFF = 1  # the absolute difference between chronological and predicted ages
AGE_FILTER = False  # using patients with this chronological age.
NORMAL_ECG = True

KEEP_AGE = False  # if False, use one of the following options of the age to reconstruct the ECGs

DATA_OUTPUT_DIR = f'output/p{FILE_NUM}'

PREDICTED_AGE_CSV = f'{DATA_OUTPUT_DIR}/prediction.csv'

batch_size = 8
traces_dset = 'tracings'
N_LEADS = 12
SAVE_HDF5 = False
