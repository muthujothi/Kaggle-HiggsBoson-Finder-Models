import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('C:/LearningMaterials/Kaggle/HiggsBoson/training.csv')

df_signals = df[df['Label'] == 's']
print df_signals.shape

df_noise = df[df['Label'] == 'b']
print df_noise.shape

signals_mass = df_signals.ix[:,1]
#print np.mean(signals_mass)

noise_mass = df_noise.ix[:, 1]
#plt.hist(signals_mass, 100)
#plt.show()

print np.percentile(signals_mass, 50)
print np.percentile(noise_mass, 50)
