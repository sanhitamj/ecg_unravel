import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append("../")

from evaluate_script import calculate_removal_error

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

# intervals = [x for x in range(10, 26)]

intervals = [10, 11, 12, 20, 21, 22, 23, 24, 25]


def plot_removal_rmse(df_err, traces, pixels, n_subjects):
    start_pixel = df_err['start_pixel'].min()
    end_pixel = df_err['start_pixel'].max()
    p = [x for x in range(start_pixel, end_pixel)]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Start Removal Pixel')
    ax1.set_ylabel('RMSE', color=color)
    ax1.plot(df_err['start_pixel'], df_err['rmse'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel('Averaged Beat', color=color)  # we already handled the x-label with ax1
    ax2.plot(p, traces[10, start_pixel:end_pixel, 0], color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f"{pixels} pixels removed; 1000 subjects")
    plt.savefig(f"n{n_subjects}_s{pixels}.png")
    plt.close()


if __name__ == "__main__":

    for interval in intervals:
        logger.info(f"Starting for removal interval: {interval}")
        df_err = calculate_removal_error(
            data_array_loc="../data/one_beat_array.npy",
            interval=interval,
            total_subjects=1000,
            n_idx=1,
            replace_near=True
        )

        df_err.to_csv(f"rmse_{interval}pixels_1000sub.csv", index=False)

        traces = np.load("../data/one_beat_array.npy")
        plot_removal_rmse(df_err, traces=traces, pixels=interval, n_subjects=1000)