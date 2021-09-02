### Import useful packages
import os
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

print('datasets.py: imported packages.')


### Train dataset object
class TrainDataset(Dataset):
	
	## Constructor
	def __init__(self, dataPath):

		# Get relevant frames
		self.frames = pims.ImageSequence(dataPath + '/train_set/*.tiff')

		# Get filepaths from frames
		self.names = pd.Series(self.frames._filepaths)

		# Get label filepaths from frame filepaths
		self.names = self.names.str.replace('tiff', 'txt')

		# Function to transform image/array to tensor
		self.to_tensor = transforms.ToTensor()

	## Get length of object frames
	def __len__(self):

		# Return length
		return len(self.frames)

	## Get next timen from datset
	def __getitem__(self, idx):

		# If tensor is inputted, convert it to a list
		if torch.is_tensor(idx):

			# Index conversion to list
			idx.tolist()

		# Read label as a numpy array from txt file and convert it to a tensor
		label = self.to_tensor(np.genfromtxt(self.names.iloc[idx])).to(dtype=torch.float32)

		# Read frame as a pims image and convert to a tensor
		tensor = self.to_tensor(self.frames[idx]/255).to(dtype=torch.float32)[0:3,:,:]

		# Change tensor to grayscale and add extra dimension
		tensor = ((tensor[0] + tensor[1] + tensor[2])/3).unsqueeze(0)

		# Create list of image tensor and unet mask
		sample = [tensor, label]

		# Return list
		return sample



### Test dataset object
class TestDataset(Dataset):
	
	## Constructor
	def __init__(self, dataPath):

		# Get relevant frames
		self.frames = pims.ImageSequence(dataPath + '/test_set/*.tiff')

		# Get filepaths from frames
		self.names = pd.Series(self.frames._filepaths)

		# Get label filepaths from frame filepaths
		self.names = self.names.str.replace('tiff', 'txt')

		# Function to transform image/array to tensor
		self.to_tensor = transforms.ToTensor()

	## Get length of object frames
	def __len__(self):

		# Return length
		return len(self.frames)

	## Get next timen from datset
	def __getitem__(self, idx):

		# If tensor is inputted, convert it to a list
		if torch.is_tensor(idx):

			# Index conversion to list
			idx.tolist()

		# Read label as a numpy array from txt file and convert it to a tensor
		label = self.to_tensor(np.genfromtxt(self.names.iloc[idx])).to(dtype=torch.float32)

		# Read frame as a pims image and convert to a tensor
		tensor = self.to_tensor(self.frames[idx]/255).to(dtype=torch.float32)[0:3,:,:]

		# Change tensor to grayscale and add extra dimension
		tensor = ((tensor[0] + tensor[1] + tensor[2])/3).unsqueeze(0)

		# Create list of image tensor and unet mask
		sample = [tensor, label]

		# Return list
		return sample
