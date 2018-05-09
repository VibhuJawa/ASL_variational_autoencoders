import os
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
import pickle
import sys
from config import data_dir

# Create train set
with open(os.path.join(data_dir, 'train_set.p'), 'rb') as filename:
    content = pickle.load(filename)
X_train, y_train = content['data'], content['labels']

# Create val set
with open(os.path.join(data_dir, 'val_set.p'), 'rb') as filename:
    content = pickle.load(filename)
X_val, y_val = content['data'], content['labels']

# Create val set
with open(os.path.join(data_dir, 'test_set.p'), 'rb') as filename:
    content = pickle.load(filename)
X_test, y_test = content['data'], content['labels']


# Create a preprocessor
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)


# Set up list of classifiers

classifiers = [(SGDClassifier(loss='hinge'), "Hinge Loss"), 
		(SGDClassifier(loss='log'), "Log Loss"),
		(PassiveAggressiveClassifier(), "Passive Aggressive - Hinge"),
		(AdaBoostClassifier(), "AdaBoost"),
		(Perceptron(penalty='l2'), "Perceptron L2"),
		(RandomForestClassifier(n_jobs=-1), "Random Forest"),
		(QuadraticDiscriminantAnalysis(), "QDA"),
		(GradientBoostingClassifier(), "Gradient Boosting")]

for clf, name in classifiers:
	clf.fit(X_train, y_train)
	print(name, clf.score(X_val, y_val), clf.score(X_train, y_train))
	sys.stdout.flush()
