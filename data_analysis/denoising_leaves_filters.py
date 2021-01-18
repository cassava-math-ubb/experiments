import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, io, restoration
from skimage.color import rgb2gray

def denoising_filters():
	
	cassava_leaves = io.imread('/Volumes/LaCie SSD/cassava-leaf-disease-classification/test_images/2216849948.jpg')
	cassava_leaves = rgb2gray(cassava_leaves)
	gaussian_filter_coins = filters.gaussian(cassava_leafs, sigma=2)
	med_filter_coins = filters.median(cassava_leafs, np.ones((3, 3)))
	tv_filter_coins = restoration.denoise_tv_chambolle(cassava_leafs, weight=0.1)

	plt.figure(figsize=(16, 4))
	plt.subplot(141)
	plt.imshow(cassava_leafs, cmap='gray', interpolation='nearest')
	plt.axis('off')
	plt.title('Image')
	plt.subplot(142)
	plt.imshow(gaussian_filter_leaves, cmap='gray',
	           interpolation='nearest')
	plt.axis('off')
	plt.title('Gaussian filter')
	plt.subplot(143)
	plt.imshow(med_filter_leaves, cmap='gray',
	           interpolation='nearest')
	plt.axis('off')
	plt.title('Median filter')
	plt.subplot(144)
	plt.imshow(tv_filter_leaves, cmap='gray',
	           interpolation='nearest')
	plt.axis('off')
	plt.title('TV filter')
	plt.show()

denoising_filters()