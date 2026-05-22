# Use this script along with the notebook: notebooks/error_norm.ipynb

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import ttest_ind

output_dir = "../output/removal_data_repl_interpol_6k"
plots_dir = "../output/data_removal_plots"
data_dir = "../data"

Path(plots_dir).mkdir(parents=True, exist_ok=True)

MAX_PIX = 2250
MIN_PIX = 1900

YOUNG = 40
MID = 60

def read_metadata():
    """
    Reads metadata and adds a columns of judgements based on the predicted ages
    """
    df_meta = pd.read_csv(f"{data_dir}/average_beat_metadata.csv")

    # Add predicted age labels based on the original paper
    young_filt = (df_meta['age'] - df_meta['nn_predicted_age'] >= 8)
    as_is_filt = (abs(df_meta['age'] - df_meta['nn_predicted_age']) < 8)
    old_filt = (df_meta['nn_predicted_age'] - df_meta['age'] >= 8)

    df_meta.loc[:, 'pred_label'] = None
    df_meta.loc[young_filt, 'pred_label'] = 'young'
    df_meta.loc[as_is_filt, 'pred_label'] = 'neutral'
    df_meta.loc[old_filt, 'pred_label'] = 'old'

    # add age labels based on the assumptions
    young_age = (df_meta['age'] <= YOUNG)
    mid_age = (df_meta['age'] > YOUNG) & (df_meta['age'] <= MID)
    old_age = (df_meta['age'] > MID)

    df_meta.loc[:, 'age_label'] = None
    df_meta.loc[young_age, 'age_label'] = 'young'
    df_meta.loc[mid_age, 'age_label'] = 'mid'
    df_meta.loc[old_age, 'age_label'] = 'old'
    df_meta.reset_index(inplace=True)
    return df_meta


def plot_max_error_pixel(
        pixels_removed,
        df_meta,
        chan0_avg,
        save_plot=False,
        is_age_group=False,
        use_norm_error=True,
):
    """
    Plots the maximum error pixel for each age group;
    prints p-values for different age groups.

    pixels_removed: int - number of pixels removed in the data removal experiment
    df_meta: pd.DataFrame - metadata dataframe from the original dataset
    chan0_avg: np.ndarray - average values for channel 0 across all samples
    save_plot: bool - whether to save the plot
    is_age_group: bool - whether to plot by age group or by predicted label group; if this is True, the
    function prints p-values for the comparison between age groups
    use_norm_error: bool - whether to use normalized error or absolute error for the analysis
    """

    filename = f"{output_dir}/all_subjects_and_pixels_{pixels_removed}pixels_6k_sub.csv"
    p = Path(filename)
    if p.exists():
        df = pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"{filename} not found. Run the data removal script first.")

    if use_norm_error:
        df.loc[:, 'norm_error'] = np.where(
            df['raw_prediction'] == df['replace_prediction'],
            0,
            abs(df['raw_prediction'] - df['replace_prediction']) / (df['replace_area'])
        )
    else:
        df.loc[:, 'norm_error'] = abs(df['raw_prediction'] - df['replace_prediction'])

    # Per subject what pixel gives maximum norm_error
    max_err_pixel = df.loc[
        df.groupby('subject')['norm_error'].idxmax(),
        ['subject', 'start_pixel']
    ].reset_index(drop=True)

    max_err_pixel = max_err_pixel.merge(
        df_meta[['index', 'age_label', 'pred_label']],
        left_on='subject',
        right_on='index',
        how='inner'
    ).drop(columns=['index'])

    fig, ax1 = plt.subplots()
    n_bins = MAX_PIX - MIN_PIX

    color = 'tab:red'
    ax1.set_xlabel(f'Start Pixel for Removing {pixels_removed} pixels')
    ax1.set_ylabel('Counts', color=color)
    if not is_age_group:
        ax1.hist(
            max_err_pixel['start_pixel'],
            bins=n_bins, color=color, alpha=0.5,
            label="all ages/preds"
        )
    else:
        ax1.hist(
            max_err_pixel[max_err_pixel['pred_label'] == 'young']['start_pixel'],
            bins=n_bins, color='green', alpha=0.5, label='young',
        )
        ax1.hist(
            max_err_pixel[max_err_pixel['pred_label'] == 'neutral']['start_pixel'],
            bins=n_bins, color='blue', alpha=0.2, label='neutral',
        )
        ax1.hist(
            max_err_pixel[max_err_pixel['pred_label'] == 'old']['start_pixel'],
            bins=n_bins, color='red', alpha=0.2, label='old',
        )

    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor=color)

    if isinstance(chan0_avg, np.ndarray):
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Average Chan-0 6k', color=color)  # we already handled the x-label with ax1
        pixels = [pix for pix in range(MIN_PIX, MAX_PIX)]
        ax2.plot(pixels, chan0_avg[MIN_PIX: MAX_PIX], color=color, label='Avg Channel 0')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save_plot:
        plt.savefig(f"{plots_dir}/max_pix_error_{pixels_removed}pixels_6k_sub.png")
    plt.show()

    if is_age_group:
        for comb in combinations(['young', 'neutral', 'old'], 2):
            group1 = max_err_pixel[max_err_pixel['pred_label'] == comb[0]]['start_pixel']
            group2 = max_err_pixel[max_err_pixel['pred_label'] == comb[1]]['start_pixel']
            p_value = ttest_ind(group1, group2, equal_var=False).pvalue
            print(f"P-value for {comb[0]} vs {comb[1]}: {round(p_value, 4)}")
