import os
import time
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import math
import torch.nn.init as init
import torch.distributed as dist
import torchvision.datasets as datasets
import random
import torchvision.models as models
import pandas
import PIL.Image as Image
import numpy as np
import torch.autograd.profiler as profiler
import redis
import io
from io import BytesIO
import numpy as np
from torch._utils import ExceptionWrapper
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch
import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict
from torch.utils.data.shadedataset import ShadeDataset, ShadeValDataset
## The ShadeDataset models a dataset distributed between more than one directories
## Each directory should follow ImageFolder structure, meaning that for each class,
## We should have a subfolder inside the root, under which all the images belonging to
## that specific class will be placed.

#https://github.com/rasbt/deeplearning-models	
class AlexNet(nn.Module):
	def __init__(self, num_classes):
		super(AlexNet, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), 256 * 6 * 6)
		logits = self.classifier(x)
		probas = F.softmax(logits, dim=1)
		return logits

#https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
class VGG(nn.Module):
	'''
	VGG model 
	'''
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(512, 512),
			nn.ReLU(True),
			nn.Linear(512, 10),
		)
		 # Initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				m.bias.data.zero_()


	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


def make_layers(cfg, batch_norm=False):
	layers = []
	in_channels = 3
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)

cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
		  512, 512, 512, 512, 'M'],
}
def vgg11():
	"""VGG 11-layer model (configuration "A")"""
	return VGG(make_layers(cfg['A']))


def vgg11_bn():
	"""VGG 11-layer model (configuration "A") with batch normalization"""
	return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
	"""VGG 13-layer model (configuration "B")"""
	return VGG(make_layers(cfg['B']))


def vgg13_bn():
	"""VGG 13-layer model (configuration "B") with batch normalization"""
	return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
	"""VGG 16-layer model (configuration "D")"""
	return VGG(make_layers(cfg['D']))


def vgg16_bn():
	"""VGG 16-layer model (configuration "D") with batch normalization"""
	return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
	"""VGG 19-layer model (configuration "E")"""
	return VGG(make_layers(cfg['E']))


def vgg19_bn():
	"""VGG 19-layer model (configuration 'E') with batch normalization"""
	return VGG(make_layers(cfg['E'], batch_norm=True))


#Initialization of local cache, PQ and ghost cache
red_local = redis.Redis()
PQ = heapdict.heapdict()
ghost_cache = heapdict.heapdict()
key_counter  = 0

#loss decomposition function
def loss_decompose(output, target,gpu):
	
	criterion = nn.CrossEntropyLoss(reduce = False).cuda(gpu)
	item_loss = criterion(output,target)
	loss = item_loss.mean()
	# loss -> loss needed for training
	# item_loss -> loss in item granularity. Needed for important sampling.
	return loss, item_loss 

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('train_paths', nargs="+", help="Paths to the train set")
	parser.add_argument('-master_address', type=str,
						help='The address of master node to connect into')
	parser.add_argument('-master_socket', type=str,
						help='The NCCL master socket to connect into')
	parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
						help='number of data loading workers (default: 4)')
	parser.add_argument('-g', '--gpus', default=1, type=int,
						help='number of gpus per node')
	parser.add_argument('-nr', '--nr', default=0, type=int,
						help='ranking within the nodes')
	parser.add_argument('--epochs', default=1, type=int, metavar='N',
						help='number of total epochs to run')
	parser.add_argument('--cache_training_data', default=False, action='store_true',
						help='Whether to cache the training data. Passing this option requires the system to have a redis in-memory cache installed on the localhost listening on port 6379')
	parser.add_argument('--network', default='resnet18', type=str,
						help='''The neural network we use for the training. It currently
						supports alexnet, resnet50, vgg16 and resnet18. The script will instantiate resnet18 if
						any other value is passed''')
	parser.add_argument('--batch_size', default=256, type=int, help="Batch size used for training and validation")
	parser.add_argument('--master_port', default=8888, type=int,
						help='The port on which master is listening')
	parser.add_argument('--val_paths', nargs="+", default=None, help="Path to the val set")
	parser.add_argument('-rep','--replication_factor', default=3.0, type=float, help='number of times data in cache will be repeated.')
	parser.add_argument('-wss','--working_set_size', default=0.1, type=float, help='percentage of dataset to be cached.')
	parser.add_argument('--host_ip', nargs='?', default='0.0.0.0', const='0.0.0.0', help='redis master node ip')
	parser.add_argument('--port_num', nargs='?', default=None, const='6379', help='port that redis cluster is listening to')
	parser.add_argument('-cn', '--cache_nodes', default=3, type=int,
						help='number of nodes cache is distributed across')

	args = parser.parse_args()
	args.world_size = args.gpus * args.nodes

	for i in range(len(args.train_paths)):
		args.train_paths[i] = os.path.abspath(args.train_paths[i])
	if args.val_paths is not None:
		for i in range(len(args.val_paths)):
			args.val_paths[i] = os.path.abspath(args.val_paths[i])
	print('ip: %s' % (args.master_address))
	print('port: %d' % (args.master_port))
	print('socket: %s' % (args.master_socket))
	print('epochs: %d' % (args.epochs))
	for i in range(len(args.train_paths)):
		print('train_path %d %s' %(i,args.train_paths[i]))
	if args.val_paths is not None:
		for i in range(len(args.val_paths)):
			print('val_path %d %s' % (i,args.val_paths[i]))
	print('network: %s' % (args.network))
	print('gpus: %d' % (args.gpus))
	print('batch_size: %d' % (args.batch_size))
	os.environ['MASTER_ADDR'] = args.master_address
	os.environ['MASTER_PORT'] = str(args.master_port)
	os.environ['NCCL_SOCKET_IFNAME'] = args.master_socket
	print('cache_data: {}'.format(args.cache_training_data))
	mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
	
	global PQ
	global key_id_map
	global key_counter
	global red_local
	global ghost_cache

	if args.cache_nodes > 1:
		startup_nodes = [{"host": args.host_ip, "port": args.port_num}]
		key_id_map = RedisCluster(startup_nodes=startup_nodes)
	else:
		key_id_map = redis.Redis()
	
	torch.cuda.set_device(gpu)

	rank = args.nr * args.gpus + gpu
	dist.init_process_group(backend='nccl', init_method=str.format('tcp://%s:%s' 
		% (args.master_address, str(args.master_port))), world_size=args.world_size, rank=rank)
	torch.manual_seed(0)
	if args.network == "alexnet":
		model = AlexNet(10)
		# model = torchvision.models.alexnet(pretrained=False)   
		# in_ftr = model.classifier[6].in_features
		# out_ftr = 10
		# model.classifier[6] = nn.Linear(in_ftr, out_ftr, bias=True)
		# model.cuda(gpu)
	elif args.network == "resnet50":
		model = models.resnet50(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 10
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	elif args.network == "vgg16":
		#model = models.vgg16(pretrained=False)
		model = vgg16()
	elif args.network == "googlenet":
		model = models.googlenet(pretrained=False)
	elif args.network == "resnext50":
		model = models.resnext50_32x4d(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 10
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	elif args.network == "densenet161":
		model = models.densenet161(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 10
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	elif args.network == "inceptionv3":
		model = models.resnet18(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 10
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	else:
		model = models.resnet18(pretrained=False)
		in_ftr  = model.fc.in_features
		out_ftr = 10
		model.fc = nn.Linear(in_ftr,out_ftr,bias=True)
		model = model.cuda(gpu)
	
	model.cuda(gpu)
	batch_size = args.batch_size
	# define loss function (criterion) and optimizer
	criterion = nn.CrossEntropyLoss().cuda(gpu)
	optimizer = torch.optim.SGD(model.parameters(), 0.1)
	# Wrap the model
	model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	# Data loading code
	if args.network != "alexnet":
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

	#For Alexnet
	if args.network == "alexnet":

		# Transformation 1
		transform_train = transforms.Compose([transforms.Resize((70, 70)),
										   transforms.RandomCrop((64, 64)),
										   transforms.ToTensor()
											])

		transform_test = transforms.Compose([transforms.Resize((70, 70)),
										  transforms.CenterCrop((64, 64)),
										  transforms.ToTensor()
										  ])
		# Transformation 2
		# transform_train = torchvision.transforms.Compose([
		# 	torchvision.transforms.Resize(224),
		# 	#torchvision.transforms.RandomResizedCrop((224, 224)),
		# 	torchvision.transforms.RandomHorizontalFlip(),
		# 	torchvision.transforms.ToTensor(),
		# 	torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# ])

		# transform_test = torchvision.transforms.Compose([
		# 	torchvision.transforms.Resize(224),
		# 	#torchvision.transforms.RandomResizedCrop((224, 224)),
		# 	torchvision.transforms.RandomHorizontalFlip(),
		# 	torchvision.transforms.ToTensor(),
		# 	torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		# ])


	train_dataset_locs = args.train_paths
	val_dataset_locs = args.val_paths

	setup_start = datetime.now()

	train_imagefolder = datasets.ImageFolder(train_dataset_locs[0])
	train_imagefolder_list = []
	train_imagefolder_list.append(train_imagefolder)

	train_dataset = ShadeDataset(
		train_imagefolder_list,
		transform_train,
		cache_data = args.cache_training_data,
		PQ=PQ, ghost_cache=ghost_cache, key_counter= key_counter,
		wss = args.working_set_size,
		host_ip = args.host_ip, port_num = args.port_num
		)
	if val_dataset_locs is not None:
		val_imagefolder = datasets.ImageFolder(val_dataset_locs[0])
		val_imagefolder_list = []
		val_imagefolder_list.append(val_imagefolder)

		val_dataset = ShadeValDataset(
			val_imagefolder_list,
			transform_test
			)
		val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
											batch_size=batch_size,
											shuffle=False,
											num_workers=0,
											pin_memory=True
											)

	train_sampler = torch.utils.data.shadesampler.ShadeSampler(train_dataset,
																	num_replicas=args.world_size,
																	rank=rank, batch_size=batch_size, seed = 3475785, 
																	host_ip = args.host_ip, port_num = args.port_num, rep_factor = args.replication_factor)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
											   batch_size=batch_size,
											   shuffle=False,
											   num_workers=0,
											   pin_memory=True,
											   sampler=train_sampler)

	batch_wts = []
	for j in range(batch_size):
		batch_wts.append(math.log(j+10))
	
	start = datetime.now()
	startup_latency = (start - setup_start).total_seconds()*1000000
	total_step = len(train_loader)
	batch_processing_times = []
	batch_loading_times = []
	batch_launch_to_gpu_times = []
	batch_feed_forward_times = []
	batch_backpropagation_times = []
	batch_aggregation_times = []
	iterator_initialization_latencies = []
	amount_elapsed_for_epoch = []
	epoch_total_train_loss_list = []
	val_accuracy_list = []
	val_avg_list = []
	val_diff_list = []
	total_loss_for_epoch = 0.
	detailed_log_per_epoch = pandas.DataFrame(
		{ 
		'batch_id': [],
		'batch_process_time': [],
		'batch_load_time': [],
		'batch_launch_time': [],
		'batch_feed_forward_time': [],
		'batch_backpropagation_time': [],
		'batch_aggregation_time': [],
		'batch_loss': []
		}
	)
	
	imp_indices_per_epoch = pandas.DataFrame(
		{ 
		'epoch': [],
		'imp_indices': []
		}
	)
	indices_for_process_per_epoch = pandas.DataFrame(
		{
		'epoch': [],
		'indices': []
		}

	)
	# SETTING IMP SCORES
	# imp_scores_per_epoch = pandas.DataFrame(
	# 	{ 
	# 	'epoch': [],
	# 	'imp_score': []
	# 	}
	# )
	# ---------------------------------------
	for epoch in range(args.epochs):
		model.train()
		total_loss_for_epoch = 0.
		epoch_start = datetime.now()
		# COLLECTING IMP SCORES
		# imp_scores = train_sampler.get_weights().tolist()
		# imp_scores_per_epoch = imp_scores_per_epoch.append({
		# 	'epoch': epoch,
		# 	'imp_score': imp_scores,
		# 	}
		# 	, ignore_index=True
		# )
		# ----------------------------------------
		train_sampler.set_epoch(epoch)
		batch_processing_times.append([])
		batch_launch_to_gpu_times.append([])
		batch_loading_times.append([])
		batch_feed_forward_times.append([])
		batch_backpropagation_times.append([])
		batch_aggregation_times.append([])
		batch_counter = 0
		iterator_initialization_start_time = datetime.now()
		iterator = enumerate(train_loader)
		iterator_initialization_end_time = datetime.now()
		iterator_initialization_latencies.append((iterator_initialization_end_time - iterator_initialization_start_time).total_seconds()*1000000)
		while batch_counter < len(train_loader):
			
			key_counter = red_local.dbsize()
			#estimating total number of keys in the entire cache.
			#since redis keeps almost equal number of keys in all nodes.
			key_counter = key_counter * args.cache_nodes

			batch_load_start_time = datetime.now()

			i, (images, labels, img_indices) = next(iterator)

			batch_load_end_time = datetime.now()

			batch_load_time = (batch_load_end_time - batch_load_start_time).total_seconds()*1000000

			batch_loading_times[epoch].append(batch_load_time)
		
			images = images.cuda(non_blocking=True)

			labels = labels.cuda(non_blocking=True)

			batch_launching_end_time = datetime.now()

			batch_launch_to_gpu_time = (batch_launching_end_time - batch_load_end_time).total_seconds()*1000000

			batch_launch_to_gpu_times[epoch].append(batch_launch_to_gpu_time)

		
			# Forward pass
			outputs = model(images)
			outputs = outputs.logits if args.network == "googlenet" else outputs

			loss, item_loss = loss_decompose(outputs,labels,gpu)

			batch_feed_forward_end_time = datetime.now()

			batch_feed_forward_time = (batch_feed_forward_end_time - batch_launching_end_time).total_seconds()*1000000

			batch_feed_forward_times[epoch].append(batch_feed_forward_time)

			batch_loss = loss.item()

			train_loader.sampler.pass_batch_important_scores(item_loss)
			
			total_loss_for_epoch += batch_loss

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()

			batch_backpropagation_end_time = datetime.now()

			batch_backpropagation_time = (batch_backpropagation_end_time - batch_feed_forward_end_time).total_seconds()*1000000
			batch_backpropagation_times[epoch].append(batch_backpropagation_time)

			optimizer.step()

			batch_aggregation_end_time = datetime.now()
			batch_aggregation_time = (batch_aggregation_end_time - batch_backpropagation_end_time).total_seconds()*1000000
			batch_aggregation_times[epoch].append(batch_aggregation_time)
			insertion_time = datetime.now()
			insertion_time = insertion_time.strftime("%H:%M:%S")
			sorted_img_indices = train_loader.sampler.get_sorted_index_list()
			track_batch_indx = 0

			# Updating PQ and ghost cache during training.
			if epoch > 0:
				PQ = train_dataset.get_PQ()
				ghost_cache = train_dataset.get_ghost_cache()

			for indx in sorted_img_indices:
				if key_id_map.exists(indx.item()):
					if indx.item() in PQ:
						#print("Train_index: %d Importance_Score: %f Frequency: %d Time: %s N%dG%d" %(indx.item(),batch_loss,PQ[indx.item()][1]+1,insertion_time,args.nr+1,gpu+1))
						PQ[indx.item()] = (batch_wts[track_batch_indx],PQ[indx.item()][1]+1)
						ghost_cache[indx.item()] = (batch_wts[track_batch_indx],ghost_cache[indx.item()][1]+1)
						track_batch_indx+=1
					else:
						#print("Train_index: %d Importance_Score: %f Time: %s N%dG%d" %(indx.item(),batch_loss,insertion_time,args.nr+1,gpu+1))
						PQ[indx.item()] = (batch_wts[track_batch_indx],1)
						ghost_cache[indx.item()] = (batch_wts[track_batch_indx],1)
						track_batch_indx+=1
				else:
					if indx.item() in ghost_cache:
						ghost_cache[indx.item()] = (batch_wts[track_batch_indx],ghost_cache[indx.item()][1]+1)
						track_batch_indx+=1
					else:
						ghost_cache[indx.item()] = (batch_wts[track_batch_indx],1)
						track_batch_indx+=1


			batch_process_end_time = datetime.now()
			batch_process_time = (batch_process_end_time - batch_load_start_time).total_seconds()*1000000
			batch_processing_times[epoch].append(batch_process_time)

			detailed_log_per_epoch = detailed_log_per_epoch.append({ 
				'batch_id': batch_counter + 1,
				'batch_process_time': batch_process_time,
				'batch_load_time': batch_load_time,
				'batch_launch_to_gpu_time': batch_launch_to_gpu_time,
				'batch_feed_forward_time': batch_feed_forward_time,
				'batch_backpropagation_time': batch_backpropagation_time,
				'batch_aggregation_time': batch_aggregation_time,
				'batch_loss': batch_loss
				}
				, ignore_index=True
			)

			batch_counter += 1
		
			train_dataset.set_PQ(PQ)
			train_dataset.set_ghost_cache(ghost_cache)
			train_dataset.set_num_local_samples(key_counter)

		train_loader.sampler.on_epoch_end(total_loss_for_epoch/batch_counter)
		#imp_ind = train_sampler.get_idxes().tolist()
		indices_for_process = train_sampler.get_indices_for_process()
		indices_for_process_per_epoch = indices_for_process_per_epoch.append({
			'epoch': epoch,
			'indices': indices_for_process,
			}
			, ignore_index=True
		)

		# imp_indices_per_epoch = imp_indices_per_epoch.append({ 
		# 	'epoch': epoch,
		# 	'imp_indices': imp_ind,
		# 	}
		# 	, ignore_index=True
		# )
		epoch_end = datetime.now()
		amount_elapsed_for_epoch.append((epoch_end - epoch_start).total_seconds()*1000000)
		print("epoch: %d proc: %d" %(epoch,gpu))

		#Checking on validation dataset.
		model.eval()
		if val_dataset_locs is not None:
			correct = 0.
			total = 0.
			with torch.no_grad():
				for data in val_loader:
					images, labels, indices = data
					images = images.cuda(non_blocking=True)
					outputs = model(images)
					_, predicted = torch.max(outputs.to('cpu'), 1)
					c = (predicted == labels).squeeze()
					correct += c.sum().item()
					total += labels.shape[0]

			val_per = 100 * correct / total
			val_accuracy_list.append(val_per)
			if len(val_diff_list) == 0:
				val_diff_list.append(0)
			else:
				val_diff_list.append(val_per - val_accuracy_list[-2])

			val_change_avg = sum(val_diff_list[len(val_diff_list)-3:len(val_diff_list)])/3 if len(val_diff_list) >= 3 else val_diff_list[-1]
			val_avg_list.append(val_change_avg)
			train_loader.sampler.pass_curr_val_change(val_change_avg)

			val_accuracy_log = pandas.DataFrame(
				{
					'val_accuracy': val_accuracy_list
				}
			)
			epoch_time_log = pandas.DataFrame(
				{
					'epoch_time': amount_elapsed_for_epoch
				}
			)
			val_diff_log = pandas.DataFrame(
				{
					'val_diff': val_diff_list
				}
			)
			val_avg_log = pandas.DataFrame(
				{
					'val_avg': val_avg_list
				}
			)
			val_accuracy_log.to_csv("val_accuracy_log"+ str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
			epoch_time_log.to_csv("epoch_time_log"+ str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
			val_diff_log.to_csv("val_diff_log"+ str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
			val_avg_log.to_csv("val_avg_log"+ str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')


		epoch_total_train_loss_list.append(total_loss_for_epoch)

	training_time = sum(amount_elapsed_for_epoch)

	process_share = len(train_loader.sampler)
	batch_processing_avg_metrics = pandas.DataFrame(
		{ 
		'batch_process_time': [ np.sum(batch_processing_times[epoch])  / len(train_loader) for epoch in range(args.epochs) ],
		'batch_load_time': [ np.sum(batch_loading_times[epoch]) / len(train_loader) for epoch in range(args.epochs) ],
		'batch_launch_time': [np.sum(batch_launch_to_gpu_times[epoch]) / len(train_loader) for epoch in range(args.epochs)],
		'batch_feed_forward_time': [np.sum(batch_feed_forward_times[epoch]) / len(train_loader) for epoch in range(args.epochs)],
		'batch_backpropagation_time': [np.sum(batch_backpropagation_times[epoch]) / len(train_loader) for epoch in range(args.epochs)],
		'batch_aggregation_time': [np.sum(batch_aggregation_times[epoch]) / len(train_loader) for epoch in range(args.epochs)],
		'iterator_loading_time_avg_time': [np.sum(iterator_initialization_latencies) / len(train_loader) for epoch in range(args.epochs)],
		'iterator_loading_time_max_time': [np.max(iterator_initialization_latencies) for epoch in range(args.epochs)],
		'total_time_elapsed': [ amount_elapsed_for_epoch[epoch] for epoch in range(args.epochs) ],
		'total_train_loss': epoch_total_train_loss_list
		}
	)
	cifar10_size = len(train_dataset)
	batch_processing_avg_metrics.to_csv('log_' + str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
	detailed_log_per_epoch.to_csv('detailed_log_' + str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
	main_logfile = open("mainlog_%d_%d.json" % (args.nr + 1, gpu + 1), "w")
	main_logfile.write("{\"startup_time\": %.3f, \"training_time\": %.3f, \"cifar10_total_image_num\": %d, \"process_share\": %d}" 
		%(startup_latency, training_time, cifar10_size, process_share))
	print("GPU %d: Training completed in %.3f minutes" % (gpu, training_time / 60000000))
	#imp_indices_per_epoch.to_csv('log_imp_indices_' + str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
	indices_for_process_per_epoch.to_csv('log_indices_process_' + str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
	# SAVING IMP SCORE
	#imp_scores_per_epoch.to_csv('log_impscore_process_' + str(args.nr + 1) + "_" + str(gpu + 1) + '.csv')
	if args.cache_training_data:
		total_keys_in_redis = len(key_id_map.keys())
		total_keys_in_node = red_local.memory_stats()['keys.count']
		if args.cache_nodes == 1:
			total_keys_in_node = total_keys_in_redis
		print("Total global keys: %d Total local keys: %d" %(total_keys_in_redis, total_keys_in_node))
	main_logfile.close()
if __name__ == '__main__':

	main()
