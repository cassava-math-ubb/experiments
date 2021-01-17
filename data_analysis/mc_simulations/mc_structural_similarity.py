import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import json
import cv2
import lpips as spipl

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tabulate import tabulate

"""
	Some basic loading for the images, resulting in an ordinary Python dict with the following
	strcutre: {'image_name.jpg': {'image': <RGB-3channel>, 'label': <integer>}}
"""
class LoadImageDataset:
	def __init__(self, base_path, train_dir, test_dir, limit = 10):
		self.limit = limit
		self.imgnames = []
		self.train_dir = base_path + '/' + train_dir
		self.test_dir = base_path + '/' + test_dir
		self.labels = self._labels(base_path + '/' + 'label_num_to_disease_map.json')
		self.images = self._images(base_path + '/' + 'train.csv', self.limit)


	def _labels(self, path):
		with open(path) as f:
			data = json.load(f)
		return data


	def _images(self, csv, limit):
		train_images = {}

		f = open(csv)
		lines = f.readlines()
		for line in lines[1:self.limit]:
			line = line.strip('\n')
			bucket = line.split(',')
			bucket = [el.strip(' \n') for el in bucket]
			train_images[bucket[0]] = {
				'image': np.array(cv2.imread(self.train_dir + '/' + bucket[0])),
				'label': int(bucket[1])
			}
			self.imgnames.append(bucket[0])

		return train_images


"""
	The Image Similarity Metric conducts a Monte Carlo simulation on a randomly selected subset 
	of k images with the final purpose of determining the similarity between the images contained 
	in the training folder.

	This class makes use of the following standard metrics definitions:
		a. LPIPS: Learned Perceptual Image Patch Similarity 
		   (make sure to run: pip3 install lpips)
		b. PNSR: Peak signal-to-noise ratio
		c. SSIM: Structural similarity index measure

"""
class ISM:
	def __init__(self):
		self.lpips = 0
		self.psnr = 0
		self.ssim = 0
		self.imgpack = LoadImageDataset('/Volumes/LaCie SSD/cassava-leaf-disease-classification', 
						'train_images', 'test_images', 100)
		self._run(sample_size = 10, num_simulations = 100)


	def overview(self):
		print('Overview: ')
		print(tabulate([['LPIPS', self.lpips], 
			['PNSR', self.psnr], 
			['SSIM', self.ssim]], 
			headers=['Metric', 'Value']))


	def _run(self, sample_size, num_simulations):
		"""
			To perform an experiment, we need to compute the similarity between each
			(i, j) pair of images, and average the results. For each such pair we will
			compute all three metrics, and return the average of those individually.
		"""
		images = self._rand_subsample(sample_size)

		def generate_pairs(sample_size):
			pairs = []

			""" Generate the pairs. """
			for i in range(sample_size):
				for j in range(i, sample_size):
					pairs.append([i, j])

			return pairs

		pairs = generate_pairs(sample_size)

		trails_results = {'lpips': [], 'psnr': [], 'ssim': []}
		for i in range(num_simulations):
			l, p, s = self._simulation(pairs)
			trails_results['lpips'].append(l)
			trails_results['psnr'].append(p)
			trails_results['ssim'].append(s)

		self.lpips = np.mean(np.array(trails_results['lpips']))
		self.psnr = np.mean(np.array(trails_results['psnr']))
		self.ssim = np.mean(np.array(trails_results['ssim']))


	def _simulation(self, pairs):
		"""
			An experiment is defined as computing the similarity index for every
			pair (i, j) of images using the metrics defined above.
		"""
		experiment_results = {'lpips': [], 'psnr': [], 'ssim': []}
		for pair in pairs:
			name1 = self.imgpack.imgnames[pair[0]]
			name2 = self.imgpack.imgnames[pair[1]]
			img1 = self.imgpack.images[name1]['image']
			img2 = self.imgpack.images[name2]['image']
			
			""" call in the other module """

			l = 0
			p = 0 
			s = 0
			experiment_results['lpips'].append(l)
			experiment_results['psnr'].append(p)
			experiment_results['ssim'].append(s)

		lpips = np.mean(np.array(experiment_results['lpips']))
		psnr = np.mean(np.array(experiment_results['psnr']))
		ssim = np.mean(np.array(experiment_results['ssim']))

		return lpips, psnr, ssim


	def _rand_subsample(self, sample_size, debug = False):
		"""
			Using U, an uniformly distributed random variable, randomly select
			a number of images (sample_size) so that we can conduct a similarity
			experiemnt on them.

		"""

		if debug == True:
			print('f: _rand_subsample(' + str(sample_size) + ', True)')

		n = len(self.imgpack.imgnames)
		imgnames = []
		images = []
		debugidxs = []
		
		for u in np.random.uniform(0, n, sample_size):
			debugidxs.append(int(u))
			imgnames.append(self.imgpack.imgnames[int(u)])
		for imgname in imgnames:
			images.append(self.imgpack.images[imgname])

		if debug == True:
			print('\t| gen. subsample with idxs: ', end='')
			print(debugidxs)
		return images


def cassava_data_analysis():
	ism = ISM()
	ism.overview()


cassava_data_analysis()

