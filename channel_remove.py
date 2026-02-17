from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, r2_score

import sys
sys.path.append("../")

from constants import DATA_DIR
from evaluate_script import read_data, predict


# to predict for the first 100 patients from the file 16
n_total = 100  # to use filters; use a positive number to use first n
batch_size = 8


def remove_single_chan():
    outdir_1 = f"{outdir}/one_chan"
    Path(outdir_1).mkdir(parents=True, exist_ok=True)

    for chan in range(12):
        data_array_chan = data_array.copy()
        # replace the channel with nan
        data_array_chan[:, :, chan] = np.nan
        # replace the nans with the nanmean, take mean only for the channel axis
        data_array_chan[:, :, chan] = np.nanmean(data_array_chan, axis=2)
        # predict
        predict(
            data_array_chan,
            df[['exam_id', 'age', 'nn_predicted_age']],
            exam_ids[:len(df)],
            reconstruct=False,
            batch_size=batch_size,
            outfile=f"{outdir}/prediction_chan{chan}.csv",
            make_pred_plot=False,
        )


def remove_mult_chan(n_chan=2):
    outdir_mult = f"{outdir}/chan_{n_chan}"
    combs = [comb for comb in combinations([x for x in range(3)], r=n_chan)]

    for chans in combs:
        data_array_chan = data_array.copy()
        chan_str = '_'.join([str(chan) for chan in chans])

        # replace the channel with nan
        for chan in chans:
            data_array_chan[:, :, chan] = np.nan

        # replace the nans with the nanmean, take mean only for the channel axis
        for chan in chans:
            data_array_chan[:, :, chan] = np.nanmean(data_array_chan, axis=2)

        pred_file =f"{outdir_mult}/prediction_chan_{chan_str}.csv"

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

        assert(
            len(predictions) == len(data_array),
            "Lengths of the input traces and predictions after channel removal do not match."
        )


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
        outfile=f'{DATA_DIR}/pred_16.csv'
    )

