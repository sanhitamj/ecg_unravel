from itertools import combinations
import numpy as np
import logging
import pandas as pd
from pathlib import Path

import sys
sys.path.append("../")

from constants import DATA_DIR
from evaluate_script import read_data, predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# to predict for the first 100 patients from the file 16
n_total = 100  # to use filters; use a positive number to use first n
batch_size = 8


def remove_single_chan(outdir, data_array, df, exam_ids):
    logging.info("Starting single channel removal calculation...")
    outdir = f"{outdir}/chan_1"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    for chan in range(12):
        data_array_chan = data_array.copy()
        # replace the channel with nan
        data_array_chan[:, :, chan] = np.nan
        # replace the nans with the nanmean, take mean only for the channel axis
        data_array_chan[:, :, chan] = np.nanmean(data_array_chan, axis=2)

        pred_file = f"{outdir}/prediction_chan_{chan}.csv"

        # predict
        predict(
            data_array_chan,
            df[['exam_id', 'age', 'nn_predicted_age']],
            exam_ids[:len(df)],
            reconstruct=False,
            batch_size=batch_size,
            outfile=pred_file,
            make_pred_plot=False,
        )


def remove_mult_chan(outdir, data_array, df, exam_ids, n_chan=2):
    logging.info(f"Starting {n_chan} channels removal calculation...")

    outdir = f"{outdir}/chan_{n_chan}"
    total_channels = 12
    combs = [comb for comb in combinations([x for x in range(total_channels)], r=n_chan)]

    logging.info(f"Number of combinations possible: {len(combs)}")

    for chans in combs:
        data_array_chan = data_array.copy()
        chan_str = '_'.join([str(chan) for chan in chans])

        # replace the channel with nan
        for chan in chans:
            data_array_chan[:, :, chan] = np.nan

        # replace the nans with the nanmean, take mean only for the channel axis
        for chan in chans:
            data_array_chan[:, :, chan] = np.nanmean(data_array_chan, axis=2)

        pred_file = f"{outdir}/prediction_chan_{chan_str}.csv"

        predict(
            data_array_chan,
            df[['exam_id', 'age', 'nn_predicted_age']],
            exam_ids[:len(data_array_chan)],
            reconstruct=False,
            batch_size=batch_size,
            outfile=pred_file,
            make_pred_plot=False,
        )
        predictions = pd.read_csv(pred_file)

        assert len(predictions) == len(data_array), "Lengths of the input and predictions do not match."


def main_func(
    n_total=n_total,
    batch_size=batch_size,
):
    outdir = f"{DATA_DIR}/n_chan_removal"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Read the entire file 16; save predictions from that file to outfile
    data_array, df, exam_ids = read_data(n_total=n_total)
    predict(
        data_array,
        df,
        exam_ids,
        reconstruct=False,
        batch_size=batch_size,
        outfile=f'{DATA_DIR}/pred_16.csv',
        make_pred_plot=False
    )

    exam_ids = exam_ids
    remove_single_chan(outdir, data_array, df, exam_ids)
    logging.info("Single channel removal calculation done.")

    # remove_mult_chan(outdir, data_array, df, exam_ids, n_chan=2)


if __name__ == "__main__":
    main_func(
        n_total=1000,
        batch_size=8
    )
