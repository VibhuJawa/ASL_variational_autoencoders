import os
import numpy as np
from matplotlib.pyplot import imread

def create_y_ids(train_label_dirs):
	label_dict = {}
	for i, label in enumerate(set(train_label_dirs)):
		label_dict[label] = i
	return label_dict

def get_data(data_label_dirs, train_dir, label_dict):
	X = []
	Y = []
	for label in data_label_dirs:
	    src_path = os.path.join(train_dir, label)
	    label_images = os.listdir(src_path)
	    for image_id in label_images:
	        image_path = os.path.join(src_path,image_id)
	        np_image = imread(image_path)
	        np_image = np_image.flatten()
	        X.append(np_image)
	        Y.append(label_dict[label])
	return np.asarray(X), np.asarray(Y)


