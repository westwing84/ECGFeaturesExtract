import numpy as np
from matplotlib import pyplot as plt

path = 'C:\\Users\\ShimaLab\\Documents\\nishihara\\data\\20200709\\Komiyama\\results\\ECG\\'
features = np.load(path + 'ecg_0.npy').reshape((1, -1))
for i in range(1, 100):
    try:
        data = np.load(path + 'ecg_' + str(i) + '.npy').reshape((1, -1))
        features = np.concatenate([features, data])
    except:
        break


features_T = features.T
plt.figure()

plt.subplot(3, 3, 1)
plt.plot(features_T[0])
plt.ylabel('Mean NN [ms]')
plt.title('Mean NN')

plt.subplot(3, 3, 2)
plt.plot(features_T[2])
plt.ylabel('SDNN [ms]')
plt.title('SDNN')

plt.subplot(3, 3, 3)
plt.plot(features_T[4])
plt.ylabel('RMSSD [ms]')
plt.title('RMSSD')

plt.subplot(3, 3, 4)
plt.plot(features_T[5])
plt.ylabel('SDSD [ms]')
plt.title('SDSD')

plt.subplot(3, 3, 5)
plt.plot(features_T[8])
plt.ylabel('LF')
plt.title('Normalized powers of LF')

plt.subplot(3, 3, 6)
plt.plot(features_T[9])
plt.ylabel('HF')
plt.title('Normalized powers of HF')

plt.subplot(3, 3, 7)
plt.plot(features_T[10])
plt.ylabel('LF/HF')
plt.title('LF/HF ratio')

plt.subplot(3, 3, 8)
plt.plot(features_T[11])
plt.ylabel('Sample entropy')
plt.title('Sample entropy')

plt.subplot(3, 3, 9)
plt.plot(features_T[12])
plt.ylabel('Lyapunov exponent')
plt.title('Lyapunov exponent')

plt.tight_layout()
plt.show()
