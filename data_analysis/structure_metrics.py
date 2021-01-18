from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import lpips as spipl
import tensorflow as tf
import lpips


class LoadImageDataset:
	def __init__(self, base_path, train_dir, test_dir, limit = 10):
		self.base_path = base_path
		self.train_dir = base_path + '/' + train_dir
		self.test_dir = base_path + '/' + test_dir

		self.limit = limit
		self.imgnames = []
		self.classes = {}
		self.images = {}
		self.labels = None
		
		self._labels(base_path + '/' + 'label_num_to_disease_map.json')
		self._load_images(base_path + '/' + 'train.csv', self.limit)
		print(str(self.limit) + ' images loaded.')


	def _labels(self, path):
		with open(path) as f:
			data = json.load(f)
		self.labels = data


	def tf_open_image(self, filename):
		return tf.image.convert_image_dtype(
				tf.io.decode_jpeg(tf.io.read_file(self.base_path + '/train_images/' + filename)), 
				tf.float32)


	def _load_images(self, csv, limit):
		f = open(csv)
		lines = f.readlines()
		# !! I should load k random images and export a report file.
		for line in lines[1:self.limit]:
			line = line.strip('\n')
			bucket = line.split(',')
			bucket = [el.strip(' \n') for el in bucket]
			self.imgnames.append(bucket[0])
			self.images[bucket[0]] = {
				'image': self.tf_open_image(bucket[0]),
				'label': bucket[1]
			}


class StudyStructuralSimilarity:
	def __init__(self, imgs_limit = 50, sample_size = 5, num_simulations = 1):
		self.lpips = 0
		self.psnr = 0
		self.ssim = 0
		self.imgs_limit = imgs_limit
		self.sample_size = sample_size
		self.num_simulations = num_simulations
		self.base_path = '/Volumes/LaCie SSD/cassava-leaf-disease-classification'
		self.imgpack = LoadImageDataset(self.base_path, 'train_images', 
						'test_images', self.imgs_limit)
		self.loss_fn = lpips.LPIPS(net='alex',version='0.1')
		self.run()


	def overview(self):
		print('Overview: ')
		print(tabulate([['LPIPS', self.lpips], ['PSNR', self.psnr], ['SSIM', self.ssim]], 
			headers=['Metric', 'Value']))


	def run(self):

		def generate_pairs(sample_size):
			pairs = []
			for i in range(self.sample_size):
				for j in range(i + 1, self.sample_size):
					pairs.append([i, j])
			return pairs
		
		pairs = generate_pairs(self.sample_size)
		trails_results = {'lpips': [], 'psnr': [], 'ssim': []}
		
		for i in range(self.num_simulations):
			random_imgnames = self._rand_subsample(self.sample_size)
			l, p, s = self._simulation(pairs, random_imgnames)
			trails_results['lpips'].append(l)
			trails_results['psnr'].append(p)
			trails_results['ssim'].append(s)
			print(f'Simulation #{i + 1}: LPIPS: {l}, PSNR: {p}, SSIM: {s}')

		self.lpips = np.mean(np.array(trails_results['lpips']))
		self.psnr = np.mean(np.array(trails_results['psnr']))
		self.ssim = np.mean(np.array(trails_results['ssim']))


	def _simulation(self, pairs, random_imgnames):
		experiment_results = {'lpips': [], 'psnr': [], 'ssim': []}
		for pair in pairs:
			name_a = random_imgnames[pair[0]]
			name_b = random_imgnames[pair[1]]

			l = self.compute_lpips(name_a, name_b)
			p = self.compute_psnr(name_a, name_b)
			s = self.compute_ssim(name_a, name_b)
			experiment_results['lpips'].append(l)
			experiment_results['psnr'].append(p)
			experiment_results['ssim'].append(s)

		lpips = np.mean(np.array(experiment_results['lpips']))
		psnr = np.mean(np.array(experiment_results['psnr']))
		ssim = np.mean(np.array(experiment_results['ssim']))

		return lpips, psnr, ssim


	def compute_psnr(self, source, target):
		im1 = self.imgpack.images[source]['image']
		im2 = self.imgpack.images[target]['image']
		psnr2 = tf.image.psnr(im1, im2, max_val=1.0)
		return psnr2.numpy()


	def compute_ssim(self, source, target):
		im1 = self.imgpack.images[source]['image']
		im2 = self.imgpack.images[target]['image']
		ssim2 = tf.image.ssim(im1, im2, max_val=1.0, filter_size=11,
		                      filter_sigma=1.5, k1=0.01, k2=0.03)
		return ssim2.numpy()


	def compute_lpips(self, source, target):
		im1 = lpips.im2tensor(lpips.load_image(self.base_path + '/train_images/' + source))
		im2 = lpips.im2tensor(lpips.load_image(self.base_path + '/train_images/' + target))
		dist = self.loss_fn.forward(im1, im2)
		return dist.detach().numpy()


	def _rand_subsample(self, sample_size):
		# Random selection from the entire set of images.
		imgnames = []
		visited = []
		
		while len(imgnames) < sample_size:
			u = np.random.randint(0, self.imgs_limit - 1, 1)[0]
			if u not in visited:
				imgnames.append(self.imgpack.imgnames[u])
				visited.append(u)

		return imgnames


	def _specific_cassava_class_rand_subsample(self, sample_size, disease_id):
		# !! Random selection from the set of a given 'disease_id' class.
		pass


def cassava_data_analysis():
	# Take 10 images out of the 100 we loaded, and run 10 simulations of structural similarity 
	# in a one vs all procedure (compute a score for each individual pair and return the mean).
	study = StudyStructuralSimilarity(imgs_limit = 100, sample_size = 10, num_simulations = 10)
	study.overview()

if __name__ == '__main__':
	print('tf version: ', tf.__version__)
	cassava_data_analysis()

