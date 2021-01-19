import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from skimage import io

base_path = '/Volumes/LaCie SSD/cassava-leaf-disease-classification/'
train_path = base_path + 'train_images/'

"""
	Given the BSD has a yellow palette, we are trying to detect that RGB color space.
	
	run: python3 brown_streak_disease_color_detection.py -c 1 -s 6
"""


def open_an_image(disease_class, skip = 0):
	image_index = 0
	skidx = 0
	with open(base_path + 'train.csv') as f:
		lines = f.readlines()
		for line in lines[1:]:
			if int(line.split(',')[1]) == disease_class and skidx < skip:
				skidx += 1
			elif int(line.split(',')[1]) == disease_class and skidx == skip:
				break
			image_index += 1

	imgname = os.listdir(train_path)[image_index].strip('._')
	image = io.imread(train_path + imgname)
	return image


def color_detection(cls, num_skips):
	image = open_an_image(disease_class=cls, skip=num_skips)

	# We are playing with the RGB color space until we reach the perfect interval.
	rgb_color_limits = [
		([72, 224, 208], [183, 246, 241]),       # yellow 1: 100 <= R <= 200; 15 <= G <= 56; 17 <= B <= 50
		([17, 171, 124], [183, 246, 241]),		 # yellow 2
		([60, 200, 130], [183, 246, 241]),		 # yellow 3
	]

	outputs = []
	for (lower, upper) in rgb_color_limits:
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")

		mask = cv2.inRange(image, lower, upper)
		outputs.append(cv2.bitwise_and(image, image, mask = mask))

	plt.figure(figsize=(8, 8))
	plt.subplot(221)
	plt.imshow(image, cmap='gray', interpolation='nearest')
	plt.axis('off')
	plt.title('Image')

	plt.subplot(222)
	plt.imshow(outputs[0], cmap='gray', interpolation='nearest')
	plt.axis('off')
	plt.title('Strict detection:\n' + str(rgb_color_limits[0]))

	plt.subplot(223)
	plt.imshow(outputs[1], cmap='gray', interpolation='nearest')
	plt.axis('off')
	plt.title('Broad detection:\n' + str(rgb_color_limits[1]))

	plt.subplot(224)
	plt.imshow(outputs[2], cmap='gray',interpolation='nearest')
	plt.axis('off')
	plt.title('Manual calibration:\n' + str(rgb_color_limits[2]))

	plt.show()


arguments = argparse.ArgumentParser()
arguments.add_argument("-c", "--class", help = "path to the image")
arguments.add_argument("-s", "--skip", help = "number of images to skip")
args = vars(arguments.parse_args())

color_detection(int(args['class']), int(args['skip']))
