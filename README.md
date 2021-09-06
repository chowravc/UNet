# UNet

This Python repository implements UNets in Pytorch of two sizes.

## Training information

Create a folder `data/` and put your data set at `data/<data-set>/`.
The format of the data set is:
```
<data-set>
|-- test_set/
|-- train_set/
```
The test set and train set must contain images called `<filename>.tiff` and associated labels (numpy arrays to txt) `<filename>.txt` of the same size.
 
After creating this dataset, open `train.py`, and put path to dataset `data/<data-set>/` in line 37. Other values such as batch size, learning rate and model can be chosen here.

You can also choose the number of epochs directly in the training loop.

For the model 'UNet_small', a train image size of 172x172 is expected, and for 'UNet' it is 572x572.

Finally, run `train.py` with:
```
!python train.py
```

Training results will get stored `runs/train/<exp>/` and one weight will be stored every epoch in `checkpoints/`. The last epoch weights and best epoch weights will be stored in `weights`. An in-depth look at the training is stored to `results.txt`.

## Detection information

All images to be detected must go to new directory `data/<new-image-directory>/<your-images>.<ext>`.

After putting images here, run:
```
!python -W ignore detect.py --w <path-to-trained-weight> --src data/<your-images>/ --model <choose-model>
```

Currently, the model choices are 'UNet' and 'UNet_small'. Make sure your weight was trained for the right model.

Example weight path:
```
runs/train/exp1/weights/best.pth
```

Detection results will be stored in `runs/detect/<exp>/`, with mask representative images and numpy arrays as txt in `labels/`.
