import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import json
import cv2
import lpips as spipl

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tabulate import tabulate

print(tf.__version__)


class LoadImageDataset:
	def __init__(self, base_path, train_dir, test_dir, limit = 10):
		self.base_path = base_path
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
			train_images[bucket[1]] = {
				'image': bucket[0] # tf.io.decode_jpeg(tf.io.read_file(self.base_path + '/train_images/' + bucket[0]))
			}
			self.imgnames.append(bucket[0])

		return train_images


class StudyStructuralSimilarity:
	def __init__(self):
		self.lpips = 0
		self.psnr = 0
		self.ssim = 0
		self.imgs_limit = 50
		self.imgpack = LoadImageDataset('/Volumes/LaCie SSD/cassava-leaf-disease-classification', 
						'train_images', 'test_images', self.imgs_limit)
		self._run(sample_size = 5, num_simulations = 1)


	def overview(self):
		print('Overview: ')
		print(tabulate([['LPIPS', self.lpips], 
			['PSNR', self.psnr], 
			['SSIM', self.ssim]], 
			headers=['Metric', 'Value']))


	def _run(self, sample_size, num_simulations):
		
		def generate_pairs(sample_size):
			pairs = []

			for i in range(sample_size):
				for j in range(i + 1, sample_size):
					pairs.append([i, j])

			return pairs
		
		pairs = generate_pairs(sample_size)
		trails_results = {'lpips': [], 'psnr': [], 'ssim': []}
		
		for i in range(num_simulations):
			random_imgnames = self._rand_subsample(sample_size)
			l, p, s = self._simulation(pairs, random_imgnames)
			trails_results['lpips'].append(l)
			trails_results['psnr'].append(p)
			trails_results['ssim'].append(s)
			print('### simulation ' + str(i + 1) + ': ' + str(l) + ', ' + str(p) + ', ' + str(s))

		self.lpips = np.mean(np.array(trails_results['lpips']))
		self.psnr = np.mean(np.array(trails_results['psnr']))
		self.ssim = np.mean(np.array(trails_results['ssim']))


	def _simulation(self, pairs, random_imgnames):
		experiment_results = {'lpips': [], 'psnr': [], 'ssim': []}
		for pair in pairs:
			name_a = self.imgpack.base_path + '/train_images/' + random_imgnames[pair[0]]
			name_b = self.imgpack.base_path + '/train_images/' + random_imgnames[pair[1]]

			l = 0
			p = self.compute_psnr(name_a, name_b)
			s = self.compute_ssim(name_a, name_b)
			experiment_results['lpips'].append(l)
			experiment_results['psnr'].append(p)
			experiment_results['ssim'].append(s)

		lpips = np.mean(np.array(experiment_results['lpips']))
		psnr = np.mean(np.array(experiment_results['psnr']))
		ssim = np.mean(np.array(experiment_results['ssim']))

		return lpips, psnr, ssim


	def compute_psnr(self, name_a, name_b):
		im1 = tf.io.read_file(name_a)
		im2 = tf.io.read_file(name_b)

		im1 = tf.io.decode_jpeg(im1)
		im2 = tf.io.decode_jpeg(im2)

		# Compute PSNR over tf.uint8 Tensors.
		psnr1 = tf.image.psnr(im1, im2, max_val=255)

		# Compute PSNR over tf.float32 Tensors.
		im1 = tf.image.convert_image_dtype(im1, tf.float32)
		im2 = tf.image.convert_image_dtype(im2, tf.float32)
		psnr2 = tf.image.psnr(im1, im2, max_val=1.0)

		return psnr2.numpy()


	def compute_ssim(self, name_a, name_b):
		im1 = tf.io.read_file(name_a)
		im2 = tf.io.read_file(name_b)

		# Read images from file.
		im1 = tf.io.decode_jpeg(im1)
		im2 = tf.io.decode_jpeg(im2)

		# Compute SSIM over tf.uint8 Tensors.
		ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,
		                      filter_sigma=1.5, k1=0.01, k2=0.03)

		# Compute SSIM over tf.float32 Tensors.
		im1 = tf.image.convert_image_dtype(im1, tf.float32)
		im2 = tf.image.convert_image_dtype(im2, tf.float32)
		ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
		                      filter_sigma=1.5, k1=0.01, k2=0.03)

		return ssim2.numpy()


	def _rand_subsample(self, sample_size):
		# Takes any cassava leaf images class.
		imgnames = []
		visited = []
		
		while len(imgnames) < sample_size:
			u = np.random.randint(0, self.imgs_limit - 1, 1)[0]
			if u not in visited:
				imgnames.append(self.imgpack.imgnames[u])
				visited.append(u)

		return imgnames


	def _specific_cassava_class_rand_subsample(self, sample_size, cassava_class):
		# Takes the given cassava leaf images class into consideration.
		pass


def cassava_data_analysis():
	study = StudyStructuralSimilarity()
	study.overview()


cassava_data_analysis()

