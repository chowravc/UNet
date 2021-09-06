### Import useful packages
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims
import pathlib
import torch.optim as optim
from torch.autograd import Variable
import skimage as skm
import glob

print('model.py: imported packages.')


### Function defining double convolution layer in UNet
def double_conv(in_c, out_c):

	## Create sequential function
	conv = nn.Sequential(

		# First 2d Convolution layer
		nn.Conv2d(in_c, out_c, kernel_size = 3),

		# First single ReLU activation layer
		nn.ReLU(inplace = True),

		# Second 2d convolution layer
		nn.Conv2d(out_c, out_c, kernel_size = 3),

		# Second single ReLU activation layer
		nn.ReLU(inplace = True)
	)

	# Return double convolution
	return conv



### Function to crop tensor to dimensions of target tensor
def crop_img(tensor, target_tensor):

	## Get target tensor dimensions
	target_size = target_tensor.size()[2]

	## Get input tensor dimensions
	tensor_size = tensor.size()[2]

	## Find difference in sizes
	delta = tensor_size - target_size

	## Find how much must be trimmed from each side
	delta = delta // 2

	## Crop the input tensor and return the output
	return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]



### UNet small neural network model object: Use train/test/detect image size 172
class UNet_small(nn.Module):

	## Class constructor
	def __init__(self):

		# Initialize parent object
		super(UNet_small, self).__init__()

		# Define input image dimensions
		self.inputDims = 172

		# Define maxpooling layer
		self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		# Create first downsample double convolution layer
		self.down_conv_1 = double_conv(1, 64) # Only 1 channel at the moment (grayscale), can be changed to 3 for RGB

		# Create second downsample double convolution layer
		self.down_conv_2 = double_conv(64, 128)

		# Create third downsample double convolution layer
		self.down_conv_3 = double_conv(128, 256)

		# Define first up transpose layer
		self.up_trans_3 = nn.ConvTranspose2d(
			in_channels = 256,
			out_channels = 128,
			kernel_size = 2,
			stride = 2
		)

		# Define first upsample double convolution layer
		self.up_conv_3 = double_conv(256, 128)

		# Define second up transpose layer
		self.up_trans_4 = nn.ConvTranspose2d(
			in_channels = 128,
			out_channels = 64,
			kernel_size = 2,
			stride = 2
		)

		# Define second upsample double convolution layer
		self.up_conv_4 = double_conv(128, 64)

		# Define output single convolution layer
		self.out = nn.Conv2d(
			in_channels = 64,
			out_channels = 1,
			kernel_size = 1
		)

	## Function to generate output from model
	def forward(self, image):

		# bs, c, h, w
		# Encoder

		# Use first down double convoltion
		x1 = self.down_conv_1(image)#
		# Use maxpool
		x3 = self.max_pool_2x2(x1)

		# Use second down double convoltion
		x3 = self.down_conv_2(x3)#
		# Use maxpool
		x5 = self.max_pool_2x2(x3)

		# Use third down double convoltion
		x5 = self.down_conv_3(x5)#

		# Decoder

		# Use first up transpose
		x = self.up_trans_3(x5)
		# Crop 'x3' to correct dimensions
		y = crop_img(x3, x)
		# Use first up double convoltion
		x = self.up_conv_3(torch.cat([x, y], 1))

		# Use second up transpose
		x = self.up_trans_4(x)
		# Crop 'x1' to correct dimensions
		y = crop_img(x1, x)
		# Use second up double convoltion
		x = self.up_conv_4(torch.cat([x, y], 1))
	
		# Delete useless outputs to free RAM
		x1, x3, x5, y = None, None, None, None,

		# Use output convolution layer
		x = self.out(x)

		# Return output
		return x



### UNet neural network model object: Use train/test/detect image size 572
class UNet(nn.Module):

	## Class constructor
	def __init__(self):

		# Initialize parent object
		super(UNet, self).__init__()

		# Define input image dimensions
		self.inputDims = 572

		# Define max pool layers
		self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

		# Define first down double convolution layer
		self.down_conv_1 = double_conv(1, 64) # Only 1 channel at the moment (grayscale), can be changed to 3 for RGB

		# Define second down double convolution layer
		self.down_conv_2 = double_conv(64, 128)

		# Define third down double convolution layer
		self.down_conv_3 = double_conv(128, 256)

		# Define fourth down double convolution layer
		self.down_conv_4 = double_conv(256, 512)

		# Define fifth down double convolution layer
		self.down_conv_5 = double_conv(512, 1024)

		# Define first up transpose layer
		self.up_trans_1 = nn.ConvTranspose2d(
			in_channels = 1024,
			out_channels = 512,
			kernel_size = 2,
			stride = 2
		)

		# Define first up double convolution layer
		self.up_conv_1 = double_conv(1024, 512)

		# Define second up transpose layer
		self.up_trans_2 = nn.ConvTranspose2d(
			in_channels = 512,
			out_channels = 256,
			kernel_size = 2,
			stride = 2
		)

		# Define second up double convolution layer
		self.up_conv_2 = double_conv(512, 256)

		# Define third up transpose layer
		self.up_trans_3 = nn.ConvTranspose2d(
			in_channels = 256,
			out_channels = 128,
			kernel_size = 2,
			stride = 2
		)

		# Define third up double convolution layer
		self.up_conv_3 = double_conv(256, 128)

		# Define fourth up transpose layer
		self.up_trans_4 = nn.ConvTranspose2d(
			in_channels = 128,
			out_channels = 64,
			kernel_size = 2,
			stride = 2
		)

		# Define fourth up double convolution layer
		self.up_conv_4 = double_conv(128, 64)

		# Define output convolution layer
		self.out = nn.Conv2d(
			in_channels = 64,
			out_channels = 1,
			kernel_size = 1
		)

	## Function to generate output from model
	def forward(self, image):

		# bs, c, h, w
		# Encoder

		# Use first down double convoltion
		x1 = self.down_conv_1(image)#
		# Use maxpool
		x3 = self.max_pool_2x2(x1)

		# Use second down double convoltion
		x3 = self.down_conv_2(x3)#
		# Use maxpool
		x5 = self.max_pool_2x2(x3)

		# Use third down double convoltion
		x5 = self.down_conv_3(x5)#
		# Use maxpool
		x7 = self.max_pool_2x2(x5)

		# Use fourth down double convoltion
		x7 = self.down_conv_4(x7)#
		# Use maxpool
		x9 = self.max_pool_2x2(x7)

		# Use fifth down double convoltion
		x9 = self.down_conv_5(x9)

		# Decoder

		# Use first up transpose
		x = self.up_trans_1(x9)
		# Crop 'x7' to correct dimensions
		y = crop_img(x7, x)
		# Use first up double convoltion
		x = self.up_conv_1(torch.cat([x, y], 1))

		# Use second up transpose
		x = self.up_trans_2(x)
		# Crop 'x5' to correct dimensions
		y = crop_img(x5, x)
		# Use second up double convoltion
		x = self.up_conv_2(torch.cat([x, y], 1))

		# Use third up transpose
		x = self.up_trans_3(x)
		# Crop 'x3' to correct dimensions
		y = crop_img(x3, x)
		# Use third up double convoltion
		x = self.up_conv_3(torch.cat([x, y], 1))

		# Use fourth up transpose
		x = self.up_trans_4(x)
		# Crop 'x1' to correct dimensions
		y = crop_img(x1, x)
		# Use fourth up double convoltion
		x = self.up_conv_4(torch.cat([x, y], 1))
	
		# Delete useless outputs to free RAM
		x1, x3, x5, x7, x9, y = None, None, None, None, None, None

		# Use output convolution layer
		x = self.out(x)

		# Return output
		return x

if __name__ == '__main__':
	
	uModel = UNet()

	print(uModel)