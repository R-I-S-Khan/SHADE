import bisect
import random
import warnings

from torch._utils import _accumulate
from torch import randperm
# No 'default_generator' in torch/__init__.pyi
from torch import default_generator  # type: ignore
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from ... import Tensor, Generator

import math

import torch
from . import Sampler, Dataset
import torch.distributed as dist

import os
import time
from datetime import datetime
import argparse
#import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import random
import pandas
import PIL.Image as Image
import numpy as np
import redis
import io
from io import BytesIO
import numpy as np
from torch._utils import ExceptionWrapper
import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

#class ShadeDataset(torch.utils.data.Dataset):
class ShadeDataset(Dataset):

	def __init__(self, imagefolders, transform=None, target_transform=None, cache_data = False,
		PQ=None, ghost_cache=None, key_counter= None, 
		wss = 0.1, host_ip = '0.0.0.0', port_num = '6379'):
		_datasets = []
		self.samples = []
		self.classes = []
		self.transform = transform
		self.target_transform = target_transform
		self.loader = None
		self.cache_data = cache_data
		self.wss = wss
		for imagefolder in imagefolders:
			#dataset = torchvision.datasets.ImageFolder(root)
			dataset = imagefolder
			self.loader = dataset.loader
			_datasets.append(dataset)
			self.samples.extend(dataset.samples)
			self.classes.extend(dataset.classes)
		self.classes = list(set(self.classes))

		self.cache_portion = self.wss * len(self.samples)
		self.cache_portion = int(self.cache_portion // 1)

		if host_ip == '0.0.0.0':
			self.key_id_map = redis.Redis()
		else:
			self.startup_nodes = [{"host": host_ip, "port": port_num}]
			self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

		self.PQ = PQ
		self.ghost_cache = ghost_cache
		self.key_counter = key_counter
		
	def random_func(self):
		return 0.6858089651836363

	def set_num_local_samples(self,n):
		self.key_counter = n

	def set_PQ(self,curr_PQ):
		self.PQ = curr_PQ

	def set_ghost_cache(self,curr_ghost_cache):
		self.ghost_cache = curr_ghost_cache

	def get_PQ(self):
		return self.PQ

	def get_ghost_cache(self):
		return self.ghost_cache

	def cache_and_evict(self, path, target, index):

		if self.cache_data and self.key_id_map.exists(index):
			try:
				print('hitting %d' %(index))
				byte_image = self.key_id_map.get(index)
				byteImgIO = io.BytesIO(byte_image)
				sample = Image.open(byteImgIO)
				sample = sample.convert('RGB')
			except PIL.UnidentifiedImageError:
				try:
					print("Could not open image in path from byteIO: ", path)
					sample = Image.open(path)
					sample = sample.convert('RGB')
					print("Successfully opened file from path using open.")
				except:
					print("Could not open even from path. The image file is corrupted.")
		else:
			if index in self.ghost_cache:
				print('miss %d' %(index))
			image = Image.open(path)
			keys_cnt = self.key_counter + 50

			if(keys_cnt >= self.cache_portion):
				try:
					peek_item = self.PQ.peekitem()
					if self.ghost_cache[index] > peek_item[1]: 
						evicted_item = self.PQ.popitem()
						print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

						if self.key_id_map.exists(evicted_item[0]):
							self.key_id_map.delete(evicted_item[0])
						keys_cnt-=1
				except:
					print("Could not evict item or PQ was empty.")
					pass

			if self.cache_data and keys_cnt < self.cache_portion:
				byte_stream = io.BytesIO()
				image.save(byte_stream,format=image.format)
				byte_stream.seek(0)
				byte_image = byte_stream.read()
				self.key_id_map.set(index, byte_image)
				print("Index: ", index)
			sample = image.convert('RGB')
		return sample

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (sample, target, index) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		insertion_time = datetime.now()
		insertion_time = insertion_time.strftime("%H:%M:%S")
		print("train_search_index: %d time: %s" %(index, insertion_time))

		sample = self.cache_and_evict(path,target,index)

		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, index

	def __len__(self) -> int:
		return len(self.samples)


class ShadeValDataset(Dataset):

	def __init__(self, imagefolders, transform=None, target_transform=None, cache_data = False):
		_datasets = []
		self.samples = []
		self.classes = []
		self.transform = transform
		self.target_transform = target_transform
		self.loader = None
		self.cache_data = cache_data

		for imagefolder in imagefolders:
			dataset = imagefolder
			self.loader = dataset.loader
			_datasets.append(dataset)
			self.samples.extend(dataset.samples)
			self.classes.extend(dataset.classes)
		self.classes = list(set(self.classes))

	def random_func(self):
		return 0.6858089651836363

	def __getitem__(self, index: int):
		"""
		Args:
			index (int): Index
		Returns:
			tuple: (sample, target, index) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		
		image = Image.open(path)
		sample = image.convert('RGB')

		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return sample, target, index

	def __len__(self) -> int:
		return len(self.samples)
