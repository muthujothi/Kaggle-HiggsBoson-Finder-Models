import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.preprocessing import Imputer

#Let us make it more pythonic.
train_data = pd.read_csv('C:/LearningMaterials/Kaggle/HiggsBoson/training_jet0.csv')
x_train = train_data.values[:, 1:19] #.values will convert the dataframe into a numpy array and in that use the two dimensional python slice notation
y_train = train_data.values[:, 20]
#scaler_x = preprocessing.StandardScaler().fit(x_train)
#x_train = scaler_x.transform(x_train)

# Train the GradientBoostingClassifier using our good features
print 'Training classifier (this may take some time!)'
gbc = GBC(n_estimators=50, max_depth=5,min_samples_leaf=200,max_features=10,verbose=1)
#gbc = linear_model.LogisticRegression()
gbc.fit(x_train,y_train) 

# Get the probaility output from the trained method, using the 10% for testing
prob_predict_train = gbc.predict_proba(x_train)[:,1]

# Experience shows me that choosing the top 15% as signal gives a good AMS score.
# This can be optimized though!
print "chossing cut off probability to predict an event as a signal..."
pcut = np.percentile(prob_predict_train,90)
print pcut

print "starting in test data"
data = pd.read_csv("C:/LearningMaterials/Kaggle/HiggsBoson/test_jet0.csv")
X_test = data.values[:, 1:]
#X_test = scaler_x.transform(X_test)

ids = data.EventId

d = gbc.predict_proba(X_test)[:, 1]

#r = np.argsort(d) + 1
p = np.empty(len(X_test), dtype=np.object)
p[d > pcut] = 's'
p[d <= pcut] = 'b'

df = pd.DataFrame({"EventId": ids, "Class": p, "SignalProb":d})
df.to_csv("predictions_jet0.csv", index=False, cols=["EventId", "Class", "SignalProb"])

print "done."
