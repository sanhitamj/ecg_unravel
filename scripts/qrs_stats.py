import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew

PEAK_DIST = 30
PRIM_PEAK_PROMINENCE = 0.75
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

    is_skew = False
    is_first_peak_positive = False

    if skew(signal) < 0:
        is_skew = True
        signal = -signal

    prim_peaks, _ = find_peaks(signal, prominence=(PRIM_PEAK_PROMINENCE, None))
    sec_peaks, _ = find_peaks(-signal, prominence=(SEC_PEAK_PROMINENCE, None))
    if len(sec_peaks) > 2:
        sec_peaks = [idx for idx in sec_peaks if abs(idx - prim_peaks[0]) < PEAK_DIST]
    if len(sec_peaks) > 1:
        peak_values = abs(signal[sec_peaks])
        sec_peaks = [sec_peaks[np.argmax(peak_values)]]

    # if is_skew:
    #     signal = -signal

    if len(prim_peaks) > 0:
        baseline = np.median(signal)
        min_mv = (signal[:-1] - signal[1:]).min()

        is_biphasic = len(sec_peaks) > 0 and sec_peaks[0] - prim_peaks[0] < PEAK_DIST
        if not is_biphasic:
            sec_peaks = prim_peaks

        first_peak = min(prim_peaks[0], sec_peaks[0])
        second_peak = max(prim_peaks[0], sec_peaks[0])

        if  signal[second_peak] <= signal[first_peak]:
            is_first_peak_positive = True

        # Finding the QRS onset:
        # reverse the order of the signal, for searching QRS onset on the left of the first peak:
        flipped = np.flip(signal[:first_peak])

        # find the first pixel in the flipped signal that is at or below the baseline
        if not is_first_peak_positive:
            flipped = -flipped
        # baseline_cross_idx = (flipped <= baseline).argmin() + 1
        offset_flipped = flipped[1:]
        first_minima = (flipped[:-1] <= offset_flipped).argmax()

        # find where values start declining again
        maxima_after_first_minima = (flipped[first_minima:-2] <= offset_flipped[first_minima + 1:]).argmin() + first_minima
        qrs_onset_idx = first_peak - maxima_after_first_minima
        if not is_first_peak_positive:
            qrs_onset_idx = first_peak - (first_minima + 1)

        # else:
        #     first_argmax = (flipped <= baseline).argmin()
        #     second_argmin = (flipped[first_argmax:] <= baseline).argmax()
        #     qrs_onset_idx = first_peak - (first_argmax + second_argmin) - 1 if second_argmin else None


        # To find the QRS offset:
        offset_signal = signal[(second_peak + 1) :]

        if first_peak < second_peak:
            if is_first_peak_positive:
                # find where trace gets greater than the baseline, after the second peak
                baseline_cross_idx = (signal[second_peak:] >= baseline).argmax() + second_peak
                slope_change_idx = (signal[second_peak : -1] <= offset_signal).argmin() + second_peak + 1

            else:
                baseline_cross_idx = (signal[second_peak:] <= baseline).argmax() + second_peak
                slope_change_idx = (offset_signal <= signal[second_peak : -1]).argmin() + second_peak + 1

        else:
            if not is_first_peak_positive:
                signal = -signal
            baseline_cross_idx = (signal[second_peak:] >= baseline).argmin() + second_peak
            slope_change_idx = (signal[second_peak : -1] <= offset_signal).argmax() + second_peak + 1

        # use minimum of these:
        qrs_offset_idx = min(slope_change_idx, baseline_cross_idx)

        # print(f"slope_change_idx: {slope_change_idx}, baseline_cross_idx: {baseline_cross_idx}")
        # print(f"qrs_offset_idx: {qrs_offset_idx}, second_peak: {second_peak}")
        # print (f"first second peaks: {first_peak}, {second_peak}")

        return prim_peaks[0], sec_peaks[0], qrs_onset_idx, None, qrs_offset_idx, baseline

    else:
        return None, None, None, None, None, None
