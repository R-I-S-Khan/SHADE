import torchvision
import os
import torch
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train_src', type=str, help="The path to read the compressed train dataset from")
parser.add_argument('-test_src', type=str, help="The path to read the compressed test dataset from")
parser.add_argument('-train_dest', type=str, help="The path to place the uncompressed train dataset")
parser.add_argument('-test_dest', type=str, help="The path to place the uncompressed test dataset")

args = parser.parse_args()

train_src = os.path.abspath(args.train_src)
test_src = os.path.abspath(args.test_src)
train_dest = os.path.abspath(args.train_dest)
test_dest = os.path.abspath(args.test_dest)

train_set = torchvision.datasets.CIFAR10(root=train_src, train=True,
										download=True, transform=None)
test_set = torchvision.datasets.CIFAR10(root=test_src, train=False,
										download=True, transform=None)


os.makedirs(train_dest)
os.makedirs(test_dest)

counter = 0

for data in train_set:
		img = data[0]
		label = data[1]
		label_dir = os.path.join(train_dest, str(label))
		if str(label) not in os.listdir(train_dest):
			os.mkdir(label_dir)	
		img.save(os.path.join(label_dir, "%s.jpeg" % (counter + 1)))
		counter += 1


counter = 0

for data in test_set:
		img = data[0]
		label = data[1]
		label_dir = os.path.join(test_dest, str(label))
		if str(label) not in os.listdir(test_dest):
			os.mkdir(label_dir)
		img.save(os.path.join(label_dir, "%s.jpeg" % (counter + 1)))
		counter += 1
