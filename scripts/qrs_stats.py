import numpy as np
import pandas as pd
from pathlib import Path

from scipy.signal import find_peaks
from scipy.stats import skew

PEAK_DIST = 30

# some biphasic signals have almost equal amplitudes for both peaks.
PRIM_PEAK_PROMINENCE = 0.3
SEC_PEAK_PROMINENCE = 0.15

def trim_trace(trace):
    """
    Trim the trace to remove zero padding.

    Parameters:
    trace (np.ndarray): Single channel averaged ECG trace

    Returns:
    np.ndarray: The trimmed ECG trace.
    """
    non_zero_indices = np.where(trace != 0)[0]
    if len(non_zero_indices) == 0:
        return trace  # Return original if no non-zero values found
    start = non_zero_indices[0]
    end = non_zero_indices[-1] + 1
    return trace[start:end]


def find_qrs_points(
        signal: np.ndarray
):
    """
    Find the peak of the QRS complex, even if it's a trough, treat it as a peak
    By definition:
    qrs_onset_idx < prim_peaks[0] < sec_peaks[0] < qrs_offset_idx
    """

    is_first_peak_positive = False

    if skew(signal) < 0:
        prim_peaks, _ = find_peaks(-signal, prominence=(PRIM_PEAK_PROMINENCE, None))
        sec_peaks, _ = find_peaks(signal, prominence=(SEC_PEAK_PROMINENCE, None))
    else:
        prim_peaks, _ = find_peaks(signal, prominence=(PRIM_PEAK_PROMINENCE, None))
        sec_peaks, _ = find_peaks(-signal, prominence=(SEC_PEAK_PROMINENCE, None))

    baseline = np.median(signal)

    if skew(signal):
        print(len(prim_peaks), len(sec_peaks), "skewed signal")

    #  Find QRS measurement code only if primary peaks are found
    if len(prim_peaks) > 0:
        # if multiple primary peaks, use the one with the highest amplitude
        if len(prim_peaks) > 1:
            prim_peaks = [prim_peaks[np.argmax(abs(signal[prim_peaks]))]]
        if len(sec_peaks) > 2:
            sec_peaks = [idx for idx in sec_peaks if abs(idx - prim_peaks[0]) < PEAK_DIST]
            # if there is any secondary peak, it should be within PEAK_DIST of the primary peak
            # this is to avoid detecting t-wave peak
        if len(sec_peaks) > 1:
            peak_values = abs(signal[sec_peaks])
            sec_peaks = [sec_peaks[np.argmax(peak_values)]]

        is_biphasic = len(sec_peaks) > 0 and sec_peaks[0] - prim_peaks[0] < PEAK_DIST
        if not is_biphasic:
            sec_peaks = prim_peaks

        first_peak = min(prim_peaks[0], sec_peaks[0])
        second_peak = max(prim_peaks[0], sec_peaks[0])

        if signal[second_peak] <= signal[first_peak]:
            is_first_peak_positive = True

        # Finding the QRS onset:
        # reverse the order of the signal, for searching QRS onset on the left of the first peak:
        flipped = np.flip(signal[:first_peak])

        # find the first pixel in the flipped signal that is at or below the baseline
        if not is_first_peak_positive:
            flipped = -flipped
        offset_flipped = flipped[1:]
        first_minima = (flipped[:-1] <= offset_flipped).argmax()

        # find where values start declining again
        maxima_after_first_minima = (flipped[first_minima:-2] <= offset_flipped[first_minima + 1:]).argmin() + first_minima
        qrs_onset_idx = first_peak - maxima_after_first_minima
        if not is_first_peak_positive:
            qrs_onset_idx = first_peak - first_minima

        # To find the QRS offset:
        offset_signal = signal[(second_peak + 1) :]

        if first_peak != second_peak:
            if is_first_peak_positive:
                # find where trace gets greater than the baseline, after the second peak
                baseline_cross_idx = (signal[second_peak:] >= baseline).argmax() + second_peak
                slope_change_idx = (signal[second_peak : -1] <= offset_signal).argmin() + second_peak + 1

            else:
                # find where trace voltage falls below the baseline, after the second peak
                baseline_cross_idx = (signal[second_peak:] <= baseline).argmax() + second_peak
                slope_change_idx = (offset_signal <= signal[second_peak : -1]).argmin() + second_peak + 1

        else:
            if not is_first_peak_positive:
                signal = -signal

            # find where trace gets greater than the baseline, after the second peak
            baseline_cross_idx = (signal[second_peak:] >= baseline).argmin() + second_peak
            # find where after crossing baseline, value[i] is lower than value[i+1]
            # so the trace sign changes
            slope_change_idx = (signal[second_peak : -1] <= offset_signal).argmax() + second_peak + 1

        # use minimum of these as the end of QRS complex:
        qrs_offset_idx = min(slope_change_idx, baseline_cross_idx)

        return prim_peaks[0], sec_peaks[0], qrs_onset_idx, qrs_offset_idx, baseline

    else:
        # if no peaks are found, use min-max of the signal for the amplitude
        prim_peaks = signal.argmax()
        sec_peaks = signal[signal.argmax() - PEAK_DIST: signal.argmax() + PEAK_DIST].argmin()
        return signal.argmax(), signal.argmin(), None, None, baseline


def get_all_qrs_metrics():
    """
    Run one_beat.py before running this function
    Reads all average beat files and returns indices for
    primary_peak, secondary peak (if exists, if not prim peak)
    qrs onset and qrs offset
    baseline (in mV)
    """

    metrics_beat_avg_file = "../output/qrs_metrics_avg_beat.csv"
    error_record_file = "../output/qrs_metrics_error_record.csv"

    if Path(metrics_beat_avg_file).exists():
        qrs_metrics = pd.read_csv(metrics_beat_avg_file)
        error_record = pd.read_csv(error_record_file)
    else:
        metrics_data = []
        error_subjects = []

        for file_num in range(1, 18):
            avg_beat_array = np.load(f"../data/one_beat_array_{file_num}.npy")
            for subject in range(len(avg_beat_array)):
                for channel in range(12):
                    trace = avg_beat_array[subject, :, channel]
                    signal = trim_trace(trace)
                    try:
                        prim_peak, sec_peak, qrs_onset_idx, qrs_offset_idx, baseline = find_qrs_points(signal)
                        prim_peak_ampl = signal[prim_peak] if prim_peak else np.nan
                        sec_peak_ampl = signal[sec_peak] if sec_peak else np.nan
                        metrics_data.append([
                            file_num,
                            subject,
                            channel,
                            prim_peak_ampl,
                            sec_peak_ampl,
                            baseline,
                            prim_peak,
                            sec_peak,
                            qrs_onset_idx,
                            qrs_offset_idx,
                        ])
                    except Exception as e:
                        error_subjects.append([file_num, subject, channel, e])

        qrs_metrics = pd.DataFrame(
            metrics_data,
            columns=[
                'file_num',
                'subject_idx',
                'channel',
                'prim_peak_mV',
                'sec_peak_mV',
                'baseline',
                'prim_peak_idx',
                'sec_peak_idx',
                'qrs_onset_idx',
                'qrs_offset_idx',
            ]
        ).dropna()

        error_record = pd.DataFrame(
            error_subjects,
            columns=['file_num', 'subject_idx', 'channel', 'exception']
        )

        qrs_metrics.to_csv(metrics_beat_avg_file, index=False)
        error_record.to_csv(error_record_file, index=False)
    return qrs_metrics, error_record
