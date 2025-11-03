# Save the files downloaded from zenodo in `DATA_INPUT_DIR`
DATA_INPUT_DIR = 'data'

FILE_NUM = 16  # the file number to read
# using 16 as default - that has the most data with the following 2 constraints:
# ABS_AGE_DIFF and NORMAL_ECG

ABS_AGE_DIFF = 1  # the absolute difference between chronological and predicted ages
NORMAL_ECG = True

AGE_FILTER = False  # using patients with this chronological age.

DATA_OUTPUT_DIR = f'output/p{FILE_NUM}'  # where reconstrctured ECG files would be saved.

PREDICTED_AGE_CSV = f'{DATA_OUTPUT_DIR}/prediction.csv'
# saved only if KEEP_AGE = True

batch_size = 8
traces_dset = 'tracings'  # used in reading hdf5 file.
N_LEADS = 12  # 12 channels for ECG
SAVE_HDF5 = False  # use this if working with "a lot of" patients.
EXAM_ID = 'exam_id'
