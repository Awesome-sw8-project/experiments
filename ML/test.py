# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean
from numpy import std
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd
# create dataset
#X, y = make_classification(n_samples=10, n_features=20, n_informative=15, n_redundant=5, random_state=1)
#print(y)
X = [[16, -999, 43], [26, -999, 43], [36, -999, 43], [46, -999, 43], [56, -999, 43], [66, -999, 43], [76, -999, 43], [86, -999, 43], [96, -999, 43], [106, -999, 43]]
# we use data pipeline to then retrieve the ground truth data corresponding to the train array
# format: [x,y, floor]
target_data = [[11, 4, 7], [2, 4, 7], [3, 4, 7], [4, 4, 7], [5, 4, 7], [6, 4, 7], [7, 4, 7], [8, 4, 7], [9, 4, 7], [10, 4, 7]]
target_dataframe = np.array(target_data)

target_x = target_dataframe[:,0]
target_y = target_dataframe[:,1]
target_floor = target_dataframe[:,2]





# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = LinearRegression()
#model1 = LogisticRegression()
#model2 = LogisticRegression()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#scores1 = cross_val_score(model1, X, target_y, scoring='accuracy', cv=cv, n_jobs=-1)
#scores2 = cross_val_score(model2, X, target_floor, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
#print('Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))
#print('Accuracy: %.3f (%.3f)' % (mean(scores2), std(scores2)))