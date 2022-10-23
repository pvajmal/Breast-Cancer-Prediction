#import Libraries

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pprint 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import sys


#Read Data
data = pd.read_csv('breast-cancer.csv')

data.drop('id', axis =1, inplace= True)
data['diagnosis'] = (data['diagnosis'] == 'M').astype(int)
#print(data.shape)

#Get highly correlated features
corr = data.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, cmap='viridis', annot=True)
plt.savefig('correlation.png')


# Get the absolute value of the correlation
cor_target = abs(corr["diagnosis"])


# Select highly correlated features (thresold = 0.2)
relevant_features = cor_target[cor_target>0.2]

# Collect the names of the features
names = [index for index, value in relevant_features.iteritems()]
#print(names)
# Drop the target variable from the results
names.remove('diagnosis')
# Display the results
#pprint.pprint(names)

X = data[names]
#print(X.shape)
Y = data['diagnosis']
y = data['diagnosis'].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 41)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#Model training
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

#Model testing
y_pred = model.predict(X_test)
print(len(y_pred))
accuracy = accuracy_score(Y_test, y_pred)
print(accuracy)

