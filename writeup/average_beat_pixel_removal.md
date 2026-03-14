# Title
### Authors

## Abstract

## Introduction

## Age Prediction in Frequency Domain

The analysis was initially performed using only 100 of the 20,000 subjects from part 16 of the available 17 datasets. For each of the 100 subjects, a notch pass filter was applied to all 12 channels of their raw ECG data. The notch pass was implemented at frequencies ranging from 2Hz to 50Hz with a quality factor of 20.0. The notch pass was implemented using the `signal.iirnotch` method from the `scipy` library.

For each of the frequencies for which a notch pass was implemented, predictions of chronological age were obtained for each of the 10,000 subjects. These predictions were then compared to the predictions from the model on the original data using mean squared error (MSE). Let $i$ index subject and $f$ index notch pass frequency. Then, the MSE was calculated for each frequency as

$$MSE_f = \sum_i (\hat{y_{if}}-\hat{y_i'})^2/n$$

where $\hat{y_{if}}$ is the prediction for subject $i$ with notch pass frequency $f$, $\hat{y_i'}$ is the prediction for subject $i$ with no notch pass implemented (i.e., the predicted chronological age in the original research study), and $n$ is the number of subjects. For frequencies which carry more useful information for predicting age, MSE will be larger since omitting those frequencies from the raw data will result in a greater change in the predicted age.

The calculated MSE is shown against the notch pass frequency in the following figure. The peaks are at 7Hz and 17-19Hz.

Link to be updated before merge.
![MSE vs notch pass frequency](https://github.com/sanhitamj/ecg_unravel/blob/avg_pix_remv_write/output/images/frequency_vs_mse.png?raw=true)

## Averaging Heart Beats into One

To find what part of the ECG contains the most information pertaining to age prediction, we used the following steps:

1. Limited the number of subject depending upon the quality of the data
2. Averaged their heartbeats into a single beat
3. Removed certain parts of the averaged beat
4. Used this as the input for the neural network to predict age, to find the RMSE

### Subject Selection

For the ease of handling the data, we used only one file, Part 16, from the validation set the authors *reference* have published. For each of the 20,000 subjects in the file, we used the following filters to select good data.

* Found the number QRS peaks from each channel, for each subject. We used `find_peaks` from `scipy`library for this purpose. In this case, peaks includes both peaks in the positive and negative directions.
* Peaks are only counted if the vertical difference between the peak and the nearest trough (the `prominence` parameter in `find_peaks`) is at least 0.75.
* The standard deviation of the interbeat interval must be relatively low. Specifically, the standard deviation of interbeat intervals must be below the $90^{th}$ percentile of such interbeat intervals. The interbeat intervals are calculated using only the channels where the number of observed peaks is the mode number of peaks across all channels for a single subject.
* The number of peaks for at least 9 out of 12 channels have to be the same for the subject to be included.
* Heart rate of each included subject has to be between 50 and 120 beats per minute.

For file #16, we found 6044 subjects that match these criteria. Using the young/old criteria from the *reference*, we have used the labels as following:

| Criteria                     | label     | Count |
| ---------------------------- | --------- | ----- |
| age - predicted age >= 8     | ECG Young | 1283  |
| abs(age - predicted age) < 8 | Neutral   | 3403  |
| predicted age - age >= 8     | ECG Old   | 1358  |

[Chronological Age VS Predicted Age for the sample used in Averaged Heartbeat](https://github.com/sanhitamj/ecg_unravel/blob/avg_pix_remv_write/writeup/selected_subjects_age_vs_orig_predicted.png)
Figure (something) shows the distribution of their predicted age as a function of their chronological age.

### Averaging the Heart Beats

For these 6044 subjects found from the method above, we constructed an average heart beat in the following way:

* The first channel for which the number of beats is the mode number of beats was used to define the position of the QRS peak, as defined above. For example, if the mode number of QRS peaks is 10, and channel 0 has 10 beats, then we used channel 0. The distribution of channels used for 6044 subjects is shown in the following table.
channel_used

| Channel | Count of Subjects |
| ------- | ----------------- |
| 0       | 5539              |
| 1       | 479               |
| 2       | 26                |

* Using only a single channel per subject, we found the average length of the heart beat, for that subject.
* Using this length, we averaged the available heartbeats into one.
* For all the subjects, we kept the QRS peak (positive or negative) at the pixel value 2048, out of the length 4096 used in the dataset. For all subject used for this analysis, at least one channel has the QRS peak at pixel number 2048.
* For redundancy, the averaged heartbeat is slightly longer than the calculated average; we used 35% length leading to the QRS peak, and 70% after the peak.
* We filled the rest of the pixels, to make the trace length of 4096. We filled it with the first available pixel value on the left, and last available pixel value on the right.

### Age Prediction for the Averaged Beat

Figure *reference* shows the comparison of the predicted ages after averaging heartbeats as a function of the predited ages with the ECG traces as they are. The correlation coefficient between the two predictions is 0.88 (check the number). After averaging, the network keeps the trend but does not have the full range of ages.

#### Age prediction with several averaged beats

To confirm that the averaged beat has not lost any information pertaining to age prediction, we concatenated the averaged heartbeat several times. For this exercise, we used the exact averaged length of the heartbeat per subject, and removed the redundant information used in the section *reference*. As more beats are added to the trace, the range of the prediction gets higher. For example, figures *reference* show predictions for 3 and 5 averaged beats concatenated as a function of the age predictions with the traces as they are. The correlation coefficients for these two are respectively, *something* and *something*

## Prediction After Removing Data

To find what part of the ECG trace contains more information about age, we removed parts of the averaged beat and compared the prediction with that of the complete averaged beat. We expect that the error around QRS complex will be most significant because
a. that's where the biggest change in the signal happens, and also,
b. for all the subjects, the averaged beats have the QRS peak, at least for one channel at pixel 2048. Because different subjects have different heart rates, the lengths of their heartbeats will be different. So as we go away from the pixel 2048, or the QRS peak, we will be looking at different parts of the ECG for different subjects.

Using the information from section *reference* we removed 14 pixels to correspond to the frequency of *some* Hz; and 36 pixels that correspond to the frequency of *some* Hz.
