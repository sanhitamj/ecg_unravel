import h5py
from itertools import product
import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from scipy.stats import skew

from constants import (
    N_LEADS,
)


DATA_DIR = "data"

peak_prominence_for_detection = 0.75
# Minimum prominence required to retain a peak
inter_beat_sd_percentile = 0.90
# Percentile of standard deviations of inter beat SDs to set a cutoff for retention
mode_cutoff = 9
# Minimum number of channels with the same number of peaks required for retention
hr_low = 50
# Lowest HR for retention
hr_high = 120
# Highest HR for retention
PEAK_AT = 2048
# Select subject and channel to pick peaks


def create_summary_frame(data_array, exam_ids):

    # For each subject and channel, calculate the number of peaks,
    # the location of peaks,
    # and the standard deviation of the interbeat intervals.
    # This takes about 3 minutes.
    summary_frame = pd.DataFrame(
        product(range(data_array.shape[0]), range(data_array.shape[2])),
        columns=['subject', 'channel']
    )
    summary_frame['n_peaks'] = np.nan
    peak_list = []
    summary_frame['inter_beat_sd'] = np.nan
    for subject in range(data_array.shape[0]):
        if (subject + 1) % 1000 == 0:
            print('Subject ', subject + 1, ' of ', data_array.shape[0] + 1)
        for channel in range(12):
            series = data_array[subject, :, channel]
            if skew(series) < 0:
                series = -series
            peaks, _ = find_peaks(series, prominence=(peak_prominence_for_detection, None))
            n_peaks = len(peaks)
            inter_beat_sd = np.nan
            if n_peaks > 2:
                inter_beat_sd = np.std(np.diff(peaks))
            mask = (summary_frame['subject'] == subject) & (summary_frame['channel'] == channel)
            summary_frame.loc[mask, 'n_peaks'] = n_peaks
            summary_frame.loc[mask, 'inter_beat_sd'] = inter_beat_sd
            peak_list.append(peaks)
    summary_frame['peaks'] = peak_list

    # For each subject, calculate the size of the mode of the number of peaks
    modes = summary_frame.groupby('subject')['n_peaks'].value_counts().reset_index()
    modes = modes.groupby('subject').first().reset_index()

    modes.rename(columns={
        'n_peaks': 'mode_n_peaks',
        'count': 'mode_count'
        }, inplace=True)
    summary_frame = summary_frame.merge(modes[['subject', 'mode_n_peaks', 'mode_count']], how='inner', on='subject')

    # Add the heart rate from the most common peak count
    mask = summary_frame['n_peaks'] > 0
    summary_frame.loc[mask, 'min_peak'] = summary_frame.loc[mask, 'peaks'].apply(np.nanmin)
    summary_frame.loc[mask, 'max_peak'] = summary_frame.loc[mask, 'peaks'].apply(np.nanmax)
    # Get the average number of observations between peaks
    summary_frame['hr'] = (summary_frame['max_peak'] - summary_frame['min_peak']) / (summary_frame['n_peaks'] - 1)
    # Heart rate is the number of beats per 10 seconds times 6.
    summary_frame['hr'] = 6 * 4096 / summary_frame['hr']
    # Only use the ones where the number of peaks matches the mode of n_peaks
    temp = summary_frame[summary_frame['n_peaks'] == summary_frame['mode_n_peaks']].copy()
    temp = temp.groupby('subject')['hr'].mean().reset_index()
    # Drop the hr variable and merge back the subject-level summary variable.
    summary_frame.drop(columns='hr', inplace=True)
    summary_frame = summary_frame.merge(temp, on='subject', how='left')

    # Find the inter_beat_sd_percentile^th percentile of the inter_beat_sd
    inter_beat_sd_cutoff = np.nanquantile(summary_frame['inter_beat_sd'], inter_beat_sd_percentile)

    # For each subject, get the average inter beat sd using only the channels
    # where the number of peaks is the mode number of peaks.
    avg_inter_beat_sd = (
        summary_frame
        .loc[summary_frame['n_peaks'] == summary_frame['mode_n_peaks']]
        .groupby('subject')
        ['inter_beat_sd']
        .mean()
        .reset_index()
    )
    avg_inter_beat_sd.rename(columns={'inter_beat_sd': 'avg_inter_beat_sd'}, inplace=True)
    summary_frame = summary_frame.merge(avg_inter_beat_sd, on='subject', how='left')

    # Flag subjects to retain if they have at least 9 channels with the same
    # number of beats, an average inter_beat_sd on those channels less than the cutoff,
    # and a heart rate between 50 and 120
    summary_frame['retain_subject'] = (
        (summary_frame['mode_count'] >= mode_cutoff)
        & (summary_frame['avg_inter_beat_sd'] < inter_beat_sd_cutoff)
        & ((summary_frame['hr'] >= hr_low) & (summary_frame['hr'] <= hr_high))
    )

    # subject_ids for the hfd5 file
    mask_df = summary_frame.groupby(
        'subject'
    )['retain_subject'].max().reset_index()

    # Assign new IDs associated with the traces to the filtered dataframe
    subject_id_mapping = {}
    for ind, subject_id in zip(
        mask_df[mask_df['retain_subject']]['subject'].reset_index(drop=True).index,
        mask_df[mask_df['retain_subject']]['subject'].reset_index(drop=True).values):
        subject_id_mapping[subject_id] = ind

    summary_frame.loc[:, 'new_subject_id'] = summary_frame['subject'].map(subject_id_mapping)
    exam_ids = exam_ids[mask_df['retain_subject']]

    data_array = data_array[mask_df['retain_subject'], :, :]

    return summary_frame, mask_df, data_array, exam_ids


def create_average_beat(summary_frame, mask_df, data_array, exam_ids):
    # new array to store only one averaged beat
    one_beat_array = np.empty(data_array.shape)

    selected_exam_ids = list(exam_ids)
    # data_array = data_array[mask_df['retain_subject'], :, :]
    data_array_idx = [int(idx) for idx in np.where(mask_df['retain_subject'])[0]]
    start_beat = []
    end_beat = []
    channel_used = []

    # this is the index for the data_array, traces
    for data_array_index in range(len(data_array)):

        # associated dataframe index:
        subject = int(summary_frame[summary_frame['new_subject_id'] == data_array_index]['subject'].values[0])

        # The first channel that has the mode number of peaks
        channel = int(summary_frame[
            (summary_frame['subject'] == subject) &
            (summary_frame['n_peaks'] == summary_frame['mode_n_peaks'])
        ]['channel'].head(1).values[0])
        channel_used.append(channel)

        peaks = summary_frame[
            (summary_frame['subject'] == subject) &
            (summary_frame['channel'] == channel)
        ]['peaks'].values[0]

        # Use the following if the summary_df is read from csv
        # if isinstance(peaks, str):
        #     peaks = [int(item) for item in peaks.replace('[', '').replace(']', '').split()]

        # Go through all beats to create an average beat
        i = 0
        beat_length = []
        while i < len(peaks) - 1:
            beat_length.append(peaks[1 + i] - peaks[i])
            i += 1
        avg_beat_len = np.ceil(np.array(beat_length).mean())

        # just over a 1/3 of the beat to before QRS complex
        back = int(np.ceil(avg_beat_len * 0.35))
        # just over 2/3 of the beat to after  QRS complex
        forward = int(np.ceil(avg_beat_len * 0.70))

        start = PEAK_AT - back
        start_beat.append(start)
        end_beat.append(PEAK_AT + forward)

        # average the heartbearts in one beat per channel
        trace = data_array[data_array_index, :, :]

        for chan in range(12):
            beats = []
            for peak in peaks:
                if peak - back >= 0 and peak + forward < 4096:
                    one_beat = trace[int(peak - back):int(peak + forward), chan]
                    beats.append(one_beat)

            avg_one_chan = np.array(beats).mean(axis=0)
            one_beat_array[data_array_index, start: start+back + forward, chan] = avg_one_chan

    np.save(f"{DATA_DIR}/one_beat_array.npy", one_beat_array)

    # Save the metadata
    df = pd.DataFrame({
        "exam_id": selected_exam_ids,
        "data_arr_idx": data_array_idx,
        "start_beat": start_beat,
        "end_beat": end_beat,
        "channel_used": channel_used,
    })

    df.to_csv(
        f"{DATA_DIR}/average_beat_metadata.csv",
        index=False
    )


if __name__ == "__main__":

    # Read in exam metadata and limit to file 16.
    df = pd.read_csv(f'{DATA_DIR}/exams.csv')
    df = df[df['trace_file'] == 'exams_part16.hdf5']

    # Read in raw ECG data for file 16.
    filename = f"{DATA_DIR}/exams_part16.hdf5"

    with h5py.File(filename, "r") as f:
        dataset = f['tracings']
        print("Dataset shape:", dataset.shape)
        data_array = f['tracings'][()]
        exam_ids = f['exam_id'][()]

    summary_frame, mask_df, data_array, exam_ids = create_summary_frame(data_array, exam_ids)
    summary_frame.to_csv(f"{DATA_DIR}/peaks_summary.csv")
    create_average_beat(summary_frame, mask_df, data_array, exam_ids)