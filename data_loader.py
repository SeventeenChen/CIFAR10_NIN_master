# -*- coding: utf-8 -*-

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset



BATCH_SIZE = 128

##########################
### CIFAR-10 Dataset
##########################


# Note transforms.ToTensor() scales input images
# to 0-1 range
train_indices = torch.arange(0, 48000)
valid_indices = torch.arange(48000, 50000)


train_and_valid = datasets.CIFAR10(root='data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)

train_dataset = Subset(train_and_valid, train_indices)
valid_dataset = Subset(train_and_valid, valid_indices)
test_dataset = datasets.CIFAR10(root='data',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=False)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=4,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=4,
                         shuffle=True)

# # Checking the dataset
# if __name__ == '__main__':
#
# 	for images, labels in train_loader:
# 		pass
# 	print(labels[:10])
#
# 	for images, labels in train_loader:
# 		pass
# 	print(labels[:10])
#
# 	for images, labels in valid_loader:
# 		pass
# 	print(labels[:10])
#
# 	for images, labels in test_loader:
# 		pass
# 	print(labels[:10])

