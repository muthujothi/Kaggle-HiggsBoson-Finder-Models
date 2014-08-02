import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.preprocessing import Imputer

#Let us make it more pythonic.
train_data = pd.read_csv('C:/LearningMaterials/Kaggle/HiggsBoson/training.csv')
x_train = train_data.values[:, 1:31] #.values will convert the dataframe into a numpy array and in that use the two dimensional python slice notation
y_train = train_data.values[:, 32]

#Impute, transform, center/scale and then feed it to model building
#print x_train
#imp = Imputer(missing_values=int(-999), strategy='median', axis=0)
#print imp
#x_train = imp.fit_transform(x_train)
#print x_train
#250000 X 30 x_train, 250000, y_train Parametric learning will identify 30 estimates for 30 features to perform classification
#x_train = np.sqrt(1 + x_train)
#print x_train

#scaler_x = preprocessing.StandardScaler().fit(x_train)
#x_train = scaler_x.transform(x_train)


# Train the GradientBoostingClassifier using our good features
print 'Training classifier (this may take some time!)'
gbc = GBC(n_estimators=50, max_depth=5,min_samples_leaf=200,max_features=10,verbose=1)
gbc.fit(x_train,y_train) 

# Get the probaility output from the trained method, using the 10% for testing
prob_predict_train = gbc.predict_proba(x_train)[:,1]

# Experience shows me that choosing the top 15% as signal gives a good AMS score.
# This can be optimized though!
print "chossing cut off probability to predict an event as a signal..."
pcut = np.percentile(prob_predict_train,85)
print pcut

print "starting in test data"
data = pd.read_csv("C:/LearningMaterials/Kaggle/HiggsBoson/test.csv")
X_test = data.values[:, 1:]
#X_test = scaler_x.transform(X_test)

ids = data.EventId

d = gbc.predict_proba(X_test)[:, 1]

r = np.argsort(d) + 1
p = np.empty(len(X_test), dtype=np.object)
p[d > pcut] = 's'
p[d <= pcut] = 'b'

df = pd.DataFrame({"EventId": ids, "RankOrder": r, "Class": p})
df.to_csv("predictions.csv", index=False, cols=["EventId", "RankOrder", "Class"])

print "done."
