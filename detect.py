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
from PIL import Image

### Import useful scripts
from datasets import *
from model import *

print('model.py: imported packages.')


### Crop an array
def cropTen(a, b): #a<=b
	sa1, sa2 = len(a), len(a[0])
	sb1, sb2 = len(b), len(b[0])
	d1, d2 = (sb1-sa1)//2, (sb2-sa2)//2
	return b[d1:sb1-d1, d2:sb2-d2]



### Function to load image as pytorch tensor
def image_loader(image_name, loader, device):
	"""load image, returns cuda tensor"""

	## Open image path as PIL image
	image = Image.open(image_name)

	## Load image with torch loader
	image = loader(image).float()

	## Use torch autograd variable
	image = Variable(image, requires_grad=True)

	## Add extra dimension to the tensor
	image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet

	## Mount to device and return tensor
	return image.to(device)  #assumes that you're using GPU



### Function to load gray image as RGB pytorch tensor
def grey2rgb_image_loader(image_name, loader, device):
	"""load image, returns cuda tensor"""

	## Open image path as PIL image
	image = Image.open(image_name)

	## Load image with torch loader
	image = loader(image).float()

	## Use torch autograd variable
	image = Variable(image, requires_grad=True)

	## Stack up single channel as 3 RGB channels
	image = torch.stack([image[0], image[0], image[0]])

	## Add extra dimension to the tensor
	image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet

	## Mount to device and return tensor
	return image.to(device)  #assumes that you're using GPU



### Main functioning of script
if __name__ == '__main__':

	## Path to trained model
	PATH = './unet.pth'

	## Select device, CPU/GPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

	## Instantiate UNet model
	uModel = UNet_small()

	## Mount UNet to device
	uModel.to(device)

	## Load trained weights to model
	uModel.load_state_dict(torch.load(PATH))

	print('\nLoaded trained UNet.')

	## Choose image input size
	imsize = 2*(84+2)

	## Create pytorch image loader, will rescale image and crop out the center
	loader = transforms.Compose([transforms.Scale(imsize), transforms.CenterCrop((imsize,imsize)), transforms.ToTensor()])

	## Load image as tensor
	testImage = image_loader("./testImages/r3_2000.tif", loader, device)
	#testImage = grey2rgb_image_loader("./testImages/r3_1000.tif")
	#testOutput = uModel(testImage)[0][0].to("cpu").detach().numpy()

	## Run model on image and get output as numpy array
	testOutput = torch.sigmoid(5*uModel(testImage))[0][0].to("cpu").detach().numpy()

	## Create subplots
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()
	fig3, ax3 = plt.subplots()

	## Crop outut image to plot overlaid
	testImage = cropTen(testOutput, testImage[0][0].to("cpu").detach().numpy())

	## Mask the output image
	masked = np.ma.masked_where(testOutput == 1, testOutput)

	a = ax1.imshow(testImage, 'gray')
	b = ax2.imshow(testOutput)
	ax3.imshow(testImage)
	ax3.imshow(testOutput, interpolation='none', alpha=0.7)
	plt.colorbar(b)


	plt.show()