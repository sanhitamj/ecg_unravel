import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, r2_score

from evaluate_script import read_data, predict


# to predict for the first 100 patients from the file 16
n_total = 100  # to use filters; use a positive number to use first n
batch_size = 8

outdir = "data/n_chan_removal"
Path(outdir).mkdir(parents=True, exist_ok=True)


# Read the entire file 16; save predictions from that file to outfile
data_array, df, exam_ids = read_data(n_total=n_total)
predict(
    data_array,
    df,
    exam_ids,
    reconstruct=False,
    batch_size=batch_size,
    outfile='data/pred_16.csv'
)


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
            exam_ids,
            reconstruct=False,
            batch_size=batch_size,
            outfile=f"{outdir_1}/prediction_chan{chan}.csv",
            keep_orig_cols=False,
        )