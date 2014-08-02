import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model


df_1 = pd.read_csv('C:/LearningMaterials/Kaggle/HiggsBoson/training.csv', usecols=['DER_mass_transverse_met_lep', 'DER_mass_vis', 'Label'])

print "Extracting features..."
df_x1 = df_1.ix[:,0] #Heavy right skewed DER_mass_transverse_met_lep: min and max: 0.0 690.075
df_x2 = df_1.ix[:,1] #Heavy right skewed DER_mass_vis min and max: 6.329 1349.351
df_y = df_1.ix[:,2]  #Class label of the data sample. Encode the classes as 0 and 1. {b: 0, s: 1}
                     #The classfiers will emit probabilities in columns. Get the column 1 to get the probability that it it is a signal
df_x1 = df_x1.values.reshape(-1, 1)
df_x2 = df_x2.values.reshape(-1, 1)
#df_y = df_y.values.reshape(-1, 1)

'''
plt.hist(df_x1)
plt.show()
plt.hist(df_x2)
plt.show()
print df_y print np.min(df_x2), np.max(df_x2)
'''


#Take a log transform of the features {x1, x2} to resolve the skewness. Also, center and scale the feature.
print "Log transformation of the predictors to resolve the skewness"
df_x1 = np.log(1 + df_x1)
df_x2 = np.log(1 + df_x2)

print "Centering and Scaling the data"
scaler_x1 = preprocessing.StandardScaler().fit(df_x1)
scaler_x2 = preprocessing.StandardScaler().fit(df_x2)

df_x1 = scaler_x1.transform(df_x1)
df_x2 = scaler_x2.transform(df_x2)

#Prepare the data in matrix form.
print "Stack the data in matrix form"
xtrain = np.hstack((df_x1, df_x2))
ytrain = df_y


#Fit the transformed, centered, scaled data with the class labels using a logistic regression model.
print "Fitting the data to a logistic regression model."
clf = linear_model.LogisticRegression()
clf.fit(xtrain, ytrain)

#Predict the probabilities of a training instance belonging to Signal class based on the built model.
'''
print "Making predictions on the Train set itself. Later change it to a cross validation set."
pred_prob = clf.predict_proba(xtrain)[:, 1]
df_pred = pd.DataFrame({"S-Prob":pred_prob})
df_pred.to_csv("C:/LearningMaterials/Kaggle/HiggsBoson/predictions.csv", index=False, cols=["S-Prob"])
'''
print "Load the test data"
df_2 = pd.read_csv('C:/LearningMaterials/Kaggle/HiggsBoson/test.csv', usecols=['EventId', 'DER_mass_transverse_met_lep', 'DER_mass_vis'])
ids = df_2.EventId
df_x1test = df_2.ix[:,1].values.reshape(-1, 1)
df_x2test = df_2.ix[:,2].values.reshape(-1, 1)

print "Log transform the features" 
df_x1test = np.log(1 + df_x1test)
df_x2test = np.log(1 + df_x2test)

print "center and scale the features"
df_x1test = scaler_x1.transform(df_x1test)
df_x2test = scaler_x2.transform(df_x2test)

print "Prepare the test data as matrix"
xtest = np.hstack((df_x1test, df_x2test))

print "Feed it to the classifier and make a predictions"
d = clf.predict_proba(xtest)[:, 1]

print "get a sorted array of the indices of the predictin vector"
r = np.argsort(d) + 1

p = np.empty(len(xtest), dtype=np.object)

print "fill the class vector based on the heuristic chosen cut-off"
p[d >= 0.8] = 's'
p[d < 0.8] = 'b'

df = pd.DataFrame({"EventId": ids, "RankOrder": r, "Class": p})
df.to_csv("predictions.csv", index=False, cols=["EventId", "RankOrder", "Class"])

print "done."














