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

### Import useful scripts
from datasets import *
from model import *
from training_loop import *

print('model.py: imported packages.')

### Main functioning of script
if __name__ == '__main__':

	#### PART 1: Setting up train and test datasets
	print('\nPART 1: Setting up train and test datasets')

	## Setup test train split

	## Decide batch size: This is the number of train image-labels that will be fed at once.
	## Choose the largest one you can without running out of RAM.
	bs = 32

	## Path to dataset
	dataPath = 'data/defects/'

	## Instantiate the train and test datasets
	train_dataset, test_dataset = TrainDataset(), TestDataset()

	## Define the train dataset loader. This will be shuffled.
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = bs, shuffle = True)

	## Define the test dataset loader. This will not be shuffled.
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = bs, shuffle = False)

	# We then load this dataset into a dataloader-- this is what will feed the gpu. Gpu's are hungry
	# so we want to feed them as much as possible. the dataloader will spit out a list that holds multiple
	# images and labels, which can be processed in parallel in the gpu. Generally, you want the batch size to
	# be a number big enough to use up all the gpu's memory, so no resource goes to waste.

	# The axes are:
	# batch, rgb, height, width

	## Load next image-label batch
	imgs, labels = next(iter(train_loader))

	## Check if the dimensions are correct
	print('Input image tensor size:', imgs.size())
	print('Input label tensor size:', labels.size())

	print('done.')

	#### PART 2: Setting up model to train
	print('\nPART 2: Setting up model to train')

	## Instantiate model
	uModel = UNet_small()

	## Load next image-label batch
	images, labels = next(iter(train_loader))

	## Compute output from untrained model
	output = uModel(images)

	## Check if the dimensions are correct
	print('Output mask tensor size:', output.size())

	## Create plot, plotting a single output
	fig, ax = plt.subplots()
	a = ax.imshow(output[0][0].detach().numpy())
	plt.colorbar(a)

	print('Displaying test output for untrained model.')
	## Display output
	plt.show()

	print('done.')

	#### PART 3: Start training
	print('\nPart 3: Start training.')

	## Choose training device; GPU/CPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
	print(f"Training on device {device}.")

	## Choose a loss function
	loss_fn = nn.BCEWithLogitsLoss()

	## Choose the learning rate
	lr = 1e-1

	## Mount model to device
	model = uModel.to(device)

	## Instantiate optimizer
	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = .0005, momentum = .9)

	## Start training loop
	training_loop(3, optimizer, model, loss_fn, train_loader, device)
	print('\nFinished training. Saving final model.')

	## Path to final trained model
	PATH = './unet.pth'

	## Save the final trained model
	torch.save(uModel.state_dict(), PATH)