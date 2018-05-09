import os
import numpy as np
from sklearn.model_selection import train_test_split
import utils
from sklearn.linear_model import Perceptron
import sys
from sklearn.preprocessing import StandardScaler

data_dir = "/home-4/vjawa1@jhu.edu/data/data_mining_project/variational_autoencoders/data/asl-alphabet"

# Get train and test directories
train_dir = os.path.join(data_dir, 'asl_alphabet_train_compressed')
test_dir = os.path.join(data_dir, 'asl_alphabet_test_compressed')

# Get list of labels
train_label_dirs = os.listdir(train_dir)
if '.DS_Store' in train_label_dirs: train_label_dirs.remove('.DS_Store')

# Create a label dictionary to map labels to ids
label_dict = utils.create_y_ids(train_label_dirs)

# Create train set
X_train, Y_train = utils.get_data(train_label_dirs, train_dir, label_dict)

# Create splits train and dev
X_train, X_dev, y_train, y_dev = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

# Create a preprocessor
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_dev = scaler.transform(X_dev)


# Set up list of classifiers

classifiers = [(Perceptron(), "Perceptron")]
for clf, name in classifiers:
	clf.fit(X_train, y_train)
	print(name, clf.score(X_dev, y_dev))
	sys.stdout.flush()
