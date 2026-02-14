import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from evaluate_script import calculate_removal_error

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

DATA_ARRAY = "../data/one_beat_offset_1000.npy"
intervals = [x for x in range(2, 40)]

results_dir = "removal_data_repl_step"

p = Path(results_dir)
if not p.exists():
    p.mkdir(parents=True, exist_ok=True)


def plot_removal_rmse(df_err, traces, pixels, n_subjects, chan=0):
    """
    df_err: dataframe with 2 columns, start_pixel and rmse
    traces: averaged heartbeat trace
    pixels: how many pixels removed
    n_subjects: rmse calculated using as many subjects
    chan: channel used in averaged, for plotting only
    """

    start_pixel = df_err['start_pixel'].min()
    end_pixel = df_err['start_pixel'].max()
    p = [x for x in range(start_pixel, end_pixel)]

    avg_chan_0 = np.mean(traces[:, :, chan], axis=0)
    _, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Start Removal Pixel')
    ax1.set_ylabel('RMSE', color=color)
    ax1.plot(df_err['start_pixel'], df_err['rmse'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:green'
    ax2.set_ylabel(f'Averaged Beat for channel {chan}, {n_subjects} subjects', color=color)
    ax2.plot(p, avg_chan_0[start_pixel:end_pixel], color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"{pixels} pixels removed; 1000 subjects")
    plt.savefig(f"{results_dir}/n{n_subjects}_s{pixels}.png")
    # plt.show()   # don't use this in a script
    plt.close()


if __name__ == "__main__":

    for interval in intervals:
        logger.info(f"Starting for removal interval: {interval}")
        df_err = calculate_removal_error(
            data_array_loc=DATA_ARRAY,
            interval=interval,
            total_subjects=1000,
            n_idx=1,
            replace_step=True
        )

        df_err.to_csv(f"{results_dir}/rmse_{interval}pixels_1000sub.csv", index=False)

        traces = np.load(DATA_ARRAY)
        plot_removal_rmse(df_err, traces=traces, pixels=interval, n_subjects=1000)
