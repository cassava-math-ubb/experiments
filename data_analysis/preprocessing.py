from matplotlib.image import imread
import numpy as np
import os

base_path = '/Volumes/LaCie SSD/cassava-leaf-disease-classification/'
train_path = base_path + 'train_images/'
cassava_image = imread(train_path + os.listdir(train_path)[0])

print('training set size: ', len(os.listdir(train_path)))
print('one image size: ', np.product(cassava_image.shape))

path = base_path + 'train.csv'
labelsidxs = [0] * 5
with open(path) as f:
	lines = f.readlines()
	for line in lines[1:]:
		labelsidxs[int(line.split(',')[1])] += 1
print('size distribution: ', labelsidxs)
