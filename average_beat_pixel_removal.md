## Averaging Heart Beats into One

To find what part of the ECG contains the most information pertaining to age prediction, we used the following steps:

1. Limited the number of subject depending upon the quality of the data
2. Averaged their heartbeats into a single beat
3. Removed certain parts of the averaged beat
4. Used this as the input for the neural network to predict age, to find the RMSE

### Subject Selection

For the ease of handling the data, we used only one file, Part 16, from the validation set the authors *reference* have published. For each of the 20,000 subjects in the file, we used the following filters to select good data.

* Found the number QRS peaks from each channel, for each subject. We used `find_peaks` from `scipy`library for this purpose. In this case, peaks includes both crests and troughs.

* Need explanation for these 2 params -
    1. peak_prominence_for_detection = 0.75  -- parameter for find_peaks?
    2. inter_beat_sd_percentile = 0.90 -- standard deviation between the beat length should be blah-blah?

* The number of peaks for at least 9, out of 12 channels have to be the same for the subject to be included.
* Heart rate of each included subject has to be between 50 and 120 beats per minute.

For file #16, we found 6044 subjects that match these criteria. Using the young/old criteria from the *reference*, we have used the labels as following:

| Criteria                     | label     | Count |
| ---------------------------- | --------- | ----- |
| age - predicted age >= 8     | ECG Young | 1283  |
| abs(age - predicted age) < 8 | Neutral   | 3403  |
| predicted age - age >= 8     | ECG Old   | 1358  |

_selected_subjects_age_vs_orig_predicted.png_

Figure (something) shows the distribution of their predicted age as a function of their chronological age.

### Averaging the Heart Beats

For these 6044 subjects found from the method above, we constructed an average heart beat in the following way:

* The first channel for which the number of beats is the mode number of beats was used to define the position of the QRS peak, as defined above. For example, if the mode number of QRS peaks is 10, and channel 0 has 10 beats, then used channel 0. The distribution of channels used for 6044 subjects:
| Channel | Count of Subjects |
| ------- | ----------------- |
| 0       | ?                 |
| 1       | ?                 |
| 2       | ?                 |

* Using only a single channel per subject, we found the average length of the heart beat, for that subject.
* Using this length, we averaged the available heartbeats into one.
* For all the subjects, we kept the QRS peak (or trough) at the pixel value 2048, out of the length 4096 used in the dataset. For all subject used for this analysis, at least one channel has the QRS peak at pixel number 2048.
* For redundancy, the averaged heartbeat is slightly longer than the calculated average; we used 35% length leading to the QRS peak, and 70% after the peak.
* We filled the rest of the pixels, to make the trace length of 4096, we filled it with the first available pixel value on the left, and last available pixel value on the right.

### Age Prediction for the Averaged Beat

Figure (reference) shows the comparison of the predicted ages with the ECG traces as they are, and after averaging.
