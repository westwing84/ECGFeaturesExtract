from ECG.ECGFeatures import ECGFeatures
import pandas as pd
from Libs.Utils import timeToInt
from scipy import stats
import numpy as np
from Conf import Settings as set
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')

path = 'C:\\Users\\ShimaLab\\Documents\\nishihara\\data\\20200709\\Komiya\\'
path_ECGdata = path + '20200709_152213_165_HB_PW.csv'
path_EmotionTest = path + 'Komiya_M_2020_7_9_15_22_44_gameResults.csv'
path_result = path + 'result\\ECG\\normalizedFeaturesECG.csv'

data = pd.read_csv(path_ECGdata)
data_EmotionTest = pd.read_csv(path_EmotionTest)
# start = "2020-04-29 15:40:21"
# end = "2020-04-29 15:54:56"
#
# data = data.loc[(data.timestamp >= start) & (data.timestamp <= end)]

# convert timestamp to int
data.loc[:, 'timestamp'] = data.loc[:, 'timestamp'].apply(timeToInt)
data_EmotionTest.loc[:, 'Time_Start'] = data_EmotionTest.loc[:, 'Time_Start'].apply(timeToInt)
data_EmotionTest.loc[:, 'Time_End'] = data_EmotionTest.loc[:, 'Time_End'].apply(timeToInt)

data_valid = []
for idx in list(data_EmotionTest.index):
    tmp = data.loc[(data.loc[:, 'timestamp'] >= data_EmotionTest.at[idx, 'Time_Start']) & (data.loc[:, 'timestamp'] <= data_EmotionTest.at[idx, 'Time_End'])]
    if not tmp.empty:
        data_valid.append(tmp)

# features extractor
featuresExct = ECGFeatures(set.FS_ECG)
time = []
emotionTestResult = []
normalizedFeatures = []
windowsize = 60
slide = 5
i = 0

for df in data_valid:
    print(i)
    # normalize the data
    df.loc[:, 'timestamp'] = df.loc[:, 'timestamp'] - df.iat[0, 0]

    featuresEachMin = []
    t0 = 0
    tf = windowsize
    idx0 = 0
    idxf = np.where(df['timestamp'].values // 1 == tf)[0][0]
    while tf <= df.iat[-1, 0]:
        time_domain = featuresExct.extractTimeDomain(df['ecg'].values[idx0:idxf])
        freq_domain = featuresExct.extractFrequencyDomain(df['ecg'].values[idx0:idxf])
        nonlinear_domain = featuresExct.extractNonLinearDomain(df['ecg'].values[idx0:idxf])
        emotionTestResult.append(data_EmotionTest.loc[i, 'Valence':'Emotion'].values)
        featuresEachMin.append(np.concatenate([time_domain, freq_domain, nonlinear_domain]))
        time.append(np.average(df['timestamp'].values[idx0:idxf]))
        t0 += slide
        tf += slide
        if tf > df.iat[-1, 0]:
            break
        else:
            idx0 = np.where(df['timestamp'].values // 1 == t0)[0][0]
            idxf = np.where(df['timestamp'].values // 1 == tf)[0][0]

    # normalized features
    featuresEachMin = np.where(np.isnan(featuresEachMin), 0, featuresEachMin)
    featuresEachMin = np.where(np.isinf(featuresEachMin), 0, featuresEachMin)
    normalizedFeatures = np.append(normalizedFeatures, stats.zscore(featuresEachMin, 0), axis=0)

    i += 1

print(normalizedFeatures)
normalizedFeatures = np.concatenate([normalizedFeatures, emotionTestResult])

'''
# save to csv
title = ['Mean NNI', 'Number of NNI', 'SDNN', 'Mean NNI difference', 'RMSSD', 'SDSD', 'Mean heart rate',
         'Std of the heart rate series', 'Normalized powers of LF', 'Normalized powers of HF', 'LF/HF ratio',
         'Sample entropy', 'Lyapunov exponent']
df = pd.DataFrame(normalizedFeatures, columns=title)
df.to_csv(path_result)

# plot
normalizedFeatures_T = normalizedFeatures.T
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
'''
