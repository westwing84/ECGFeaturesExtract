from ECG.ECGFeatures import ECGFeatures
import pandas as pd
from Libs.Utils import timeToInt
from scipy import stats
import numpy as np
from Conf import Settings as set
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

path = 'C:\\Users\\ShimaLab\\Documents\\nishihara\\data\\20200611\\ECG\\'
path_ecgdata = path + 'hb_data_sample.csv'
path_result = path + 'result\\normalizedFeaturesECG.csv'

data = pd.read_csv(path_ecgdata)
# start = "2020-04-29 15:40:21"
# end = "2020-04-29 15:54:56"
#
# data = data.loc[(data.timestamp >= start) & (data.timestamp <= end)]

# convert timestamp to int
data["timestamp"] = data["timestamp"].apply(timeToInt)

# normalize the data
data["timestamp"] = data["timestamp"] - data.iloc[0].timestamp

# features extrator
featuresExct = ECGFeatures(set.FS_ECG)
featuresEachMin = []
time = []
windowsize = 60
slide = 5
t0 = 0
tf = windowsize

idx0 = 0
idxf = np.where(data['timestamp'].values // 1 == tf)[0][0]
while tf <= data[-2:-1].timestamp.values[0]:
    time_domain = featuresExct.extractTimeDomain(data['ecg'].values[idx0:idxf])
    freq_domain = featuresExct.extractFrequencyDomain(data['ecg'].values[idx0:idxf])
    nonlinear_domain = featuresExct.extractNonLinearDomain(data['ecg'].values[idx0:idxf])
    featuresEachMin.append(np.concatenate([time_domain, freq_domain, nonlinear_domain]))
    time.append(np.average(data['timestamp'].values[idx0:idxf]))
    t0 += slide
    tf += slide
    if tf > data[-2:-1].timestamp.values[0]:
        break
    else:
        idx0 = np.where(data['timestamp'].values // 1 == t0)[0][0]
        idxf = np.where(data['timestamp'].values // 1 == tf)[0][0]


# normalized features
featuresEachMin = np.where(np.isnan(featuresEachMin), 0, featuresEachMin)
featuresEachMin = np.where(np.isinf(featuresEachMin), 0, featuresEachMin)
normalizedFeatures = stats.zscore(featuresEachMin, 0)


# plot
normalizedFeatures_T = normalizedFeatures.T
title = ['Mean NNI', 'Number of NNI', 'SDNN', 'Mean NNI difference', 'RMSSD', 'SDSD', 'Mean heart rate',
         'Std of the heart rate series', 'Normalized powers of LF', 'Normalized powers of HF', 'LF/HF ratio',
         'Sample entropy', 'Lyapunov exponent']
num_plot = 9
if normalizedFeatures_T.shape[0] % num_plot == 0:
    num_figure = normalizedFeatures_T.shape[0] // num_plot
else:
    num_figure = normalizedFeatures_T.shape[0] // num_plot + 1

for i in range(num_figure):
    plt.figure(figsize=(12, 9))
    for j in range(num_plot):
        if num_plot*i+j >= normalizedFeatures_T.shape[0]:
            break
        plt.subplot(3, 3, j + 1)
        plt.plot(time, normalizedFeatures_T[num_plot*i+j])
        plt.xlabel('Time [s]')
        plt.title(title[num_plot*i+j])
    plt.tight_layout()
plt.show()


# np.savetxt(path_result, normalizedFeatures, delimiter=',')

