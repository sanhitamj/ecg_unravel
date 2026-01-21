import numpy as np

data_array = np.load("data/one_beat_array.npy")

n = 1000  # number of subjects to be used in sample removal exercise

data_array = data_array[:n, :, :]

for i in range(len(data_array)):
    for chan in range(12):
        arr = data_array[i, :, chan]

        try:
            idx_min = int((np.nonzero(arr))[0][0])
            begin_fill = arr[idx_min]
            idx_max = int((np.nonzero(arr))[0][-1])
            end_fill = arr[idx_max]
            data_array[i, :idx_min, chan] = begin_fill
            data_array[i, idx_max:, chan] = end_fill
        except IndexError:
            # some subjects have some channels empty
            pass

np.save("data/one_beat_offset_1000.npy", data_array)
