EXAM_ID = 'exam_id'
DATA_INPUT_DIR = 'data'

FILE_NUM = 16
ABS_AGE_DIFF = 1  # the absolute difference between chronological and predicted ages
NORMAL_ECG = True
AGE_FILTER = 20  # using patients with this chronological age.

KEEP_AGE = True  # if False, use one of the following options of the age to reconstruct the ECGs
REPLACE_AGE = 80
REPLACE_AGE_RANGE = [x for x in range(20, 80)]


DATA_OUTPUT_DIR = f'output/part{FILE_NUM}_age_diff_{ABS_AGE_DIFF}'
if AGE_FILTER:
    DATA_OUTPUT_DIR = f'{DATA_OUTPUT_DIR}_age_{AGE_FILTER}'

if not KEEP_AGE and REPLACE_AGE:
    DATA_OUTPUT_DIR = f'{DATA_OUTPUT_DIR}_replace_age_{REPLACE_AGE}'

if not KEEP_AGE and not REPLACE_AGE and REPLACE_AGE_RANGE:
    DATA_OUTPUT_DIR = f'{DATA_OUTPUT_DIR}_replace_age_range'

PREDICTED_AGE_CSV = f'{DATA_OUTPUT_DIR}/prediction.csv'
RECONSTED_ECG = f'{DATA_OUTPUT_DIR}/reconstructed.npy'

batch_size = 8
traces_dset = 'tracings'
