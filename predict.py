import pandas as pd
import numpy as np

df_2 = pd.read_csv('C:/LearningMaterials/Kaggle/HiggsBoson/test.csv', usecols=['EventId', 'DER_mass_transverse_met_lep', 'DER_mass_vis'])
ids = df_2.EventId
df_x1test = df_2.ix[:,1]
df_x2test = df_2.ix[:,2]
print ids

