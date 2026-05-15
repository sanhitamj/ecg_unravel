import json
import logging
import numpy as np
import os
import pandas as pd
import torch

from constants import (
    DATA_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    N_LEADS,
)

import sys
sys.path.append("../")

from resnet import ResNet1d

from scripts.evaluate_script import read_data

start_pct = -0.25  # The lowest percentage to start at
end_pct = 0.4  # The highest percentage to end at
window_pct = 0.05  # The size of the deletion window
by_pct = 0.025  # The increment of the start at each step
keep_only_retain_subjects = True

n_observations = 20_000

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
config = f'{MODEL_DIR}/config.json'

logger.info("Reding beats summary frame")
beats_summary = pd.read_csv(f"{DATA_DIR}/beats_summary_frame.csv")

# Instantiate the model using the config.json information.
logger.info("Instantiating model")
with open(config, 'r') as f:
    config_dict = json.load(f)
model = ResNet1d(
    input_dim=(N_LEADS, config_dict['seq_length']),
    blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])),
    n_classes=1,
    kernel_size=config_dict['kernel_size'],
    dropout_rate=config_dict['dropout_rate']
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Retrieve the state dict, which has all the coefficients
state_dict = (torch.load(f'{MODEL_DIR}/model.pth',
              weights_only=False,
              map_location=device))

# Load the state dict and set the model to eval mode.
model.load_state_dict(state_dict['model'])
model.eval()

# Read in exam metadata and limit to file 16.
logger.info("Reading exam data")
data_array, metadata, exam_ids = read_data(
    n_total=n_observations,
)

# Brute force sort df to match order of exam_ids
metadata_sorted = []
for exam_id in exam_ids:
    metadata_sorted.append(metadata[metadata['exam_id'] == exam_id])
metadata = pd.concat(metadata_sorted)
metadata['subject'] = np.arange(metadata.shape[0])

# Get information on which subjects to retain
retain_subjects = beats_summary.groupby('subject')['retain_subject'].first().reset_index()
retain_subjects = retain_subjects[:n_observations]
retain_subjects = retain_subjects[retain_subjects['retain_subject']]

if keep_only_retain_subjects:
    data_array = data_array[retain_subjects['subject'].values, :, :]
    metadata = metadata[metadata['subject'].isin(retain_subjects['subject'].values)]
    beats_summary = beats_summary[beats_summary['subject'].isin(retain_subjects['subject'].values)]

# Loop through the starting points by the increment, deleting the specified window size.
out_frames = []
for i in np.arange(start_pct, end_pct + by_pct, by_pct):
    deletion_start = i
    deletion_end = deletion_start + window_pct

    logger.info(f"Start pct {deletion_start} to end pct {deletion_end}.")

    data_array_del = data_array.copy()
    areas = []
    # Loop over all subjects
    for subject in range(data_array.shape[0]):
        if ((subject + 1) % 1_000 == 0):
            logger.info(f"Subject {subject + 1} of {data_array.shape[0]}.")

        area = 0
        subject_number = retain_subjects.iloc[subject]['subject']

        # Loop through all channels
        for channel in range(data_array.shape[2]):

            # Extract the peaks
            peaks = beats_summary.loc[(beats_summary['subject'] == subject_number)
                                      & (beats_summary['channel'] == channel), 'peaks'].values[0]
            peaks = [int(item) for item in peaks.replace("[", "").replace("]", "").split()]

            # We can only calculate beat length if there are at least 2 peaks.
            if len(peaks) >= 2:
                avg_beat_length = (peaks[-1] - peaks[0]) / (len(peaks) - 1)
                deletion_starts = [int(i + (deletion_start * avg_beat_length)) for i in peaks]
                deletion_ends = [int(i + (deletion_end * avg_beat_length)) for i in peaks]
                for i in range(len(peaks)):

                    # Deletion can't start before 0 or end after 4096
                    if (deletion_starts[i] >= 0) and (deletion_ends[i] < 4096):

                        # Replacement is a line between the first deletion point and last deletion point.
                        replace_val = (
                            np.linspace(data_array[subject, deletion_starts[i], channel],
                                        data_array[subject, deletion_ends[i], channel],
                                        deletion_ends[i] - deletion_starts[i])
                            )
                        data_array_del[subject, deletion_starts[i]:deletion_ends[i], channel] = (
                            replace_val
                            )

                        # Area deleted is the sum of the absolute values of difference between the
                        # line and the original data
                        area += float(
                            np.sum(
                                np.abs(
                                    data_array[subject,
                                               deletion_starts[i]:deletion_ends[i],
                                               channel]
                                    - replace_val)
                                )
                            )
        areas.append(area)

    # Get the predictions for data_array_del in batches
    n_total = data_array.shape[0]  # total number of predictions
    batch_size = 20
    n_batches = int(np.ceil(n_total / batch_size))

    pred_list = []
    end = 0
    for i in range(n_batches):
        if (i % 100 == 0):
            logger.info(f"Batch {i} of {n_batches}.")
        start = end
        end = min((i + 1) * batch_size, n_total)

        # Get the predictions
        model.zero_grad()
        y_pred = model(torch.tensor(data_array_del[start:end, :, :], dtype=torch.float).transpose(-1, -2))

        # Merge predictions back onto the metadata frame
        preds = pd.DataFrame({'exam_id': metadata['exam_id'][start:end],
                              'torch_pred': y_pred.detach().numpy().squeeze()})
        pred_list.append(preds)

    preds = pd.concat(pred_list, axis=0, ignore_index=True)
    compare = metadata.merge(preds, on='exam_id', how='inner')

    out_frame = pd.merge(preds, metadata[['exam_id', 'nn_predicted_age']], on='exam_id')
    out_frame['area_removed'] = areas
    out_frame['start_pct'] = deletion_start
    out_frame['end_pct'] = deletion_end

    out_frames.append(out_frame)

out_frames = pd.concat(out_frames)
os.makedirs(f'{OUTPUT_DIR}/removal_data_repl_interpol_raw', exist_ok=True)
out_frames.to_csv(f'{OUTPUT_DIR}/removal_data_repl_interpol_raw/'
                  + f'all_subjects_and_pixels_win{int(1000 * window_pct)}_inc{int(1000 * by_pct)}.csv', index=False)
