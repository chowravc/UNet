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
import datetime

### Import useful scripts
from model import *

print('training_loop.py: imported packages.')


### Function defining training loop
def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):

	## Loop through number of epochs
	for epoch in range(1, n_epochs + 1):  # <2>

		# Keep track of training loss
		loss_train = 0.0

		# Go through every image-label 'pair' from the train loader
		for imgs, labels in train_loader:  # <3>

			# Load image tensor to GPU/CPU
			imgs = imgs.to(device=device)

			# Load labels tensor to GPU/CPU
			labels = labels.to(device=device)

			# Compute outputs from the model
			outputs = model(imgs)  # <4>

			# Calculate batch loss
			loss = loss_fn(outputs, crop_img(labels, outputs))  # <5>

			# Zero out optimizer gradient
			optimizer.zero_grad()  # <6>

			# Backpropagate loss
			loss.backward()  # <7>

			# Take optimizer step
			optimizer.step()  # <8>

			# Add batch loss to total training loss
			loss_train += loss.item()  # <9>

		# Display progress every 1 epochs
		if epoch == 1 or epoch % 1 == 0:
			print('{} Epoch {}, Training loss {}, std {} , teststd {} '.format(
				datetime.datetime.now(), epoch, loss_train/len(train_loader), outputs.std(), labels.float().std()))



### Run if script is called directly
if __name__ == '__main__':
	
	## Get device GPU/CPU
	device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

	## Display found device
	print(f"Training on device {device}.")

	## Choose a loss function
	loss_fn = nn.BCEWithLogitsLoss()

	## Choose learning rate
	lr = 1e-1

	## Instantiate model
	uModel = UNet_small()

	## Mount model to device
	model = uModel.to(device)

	## Instantiate optimizer
	optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay = .0005, momentum = .9)