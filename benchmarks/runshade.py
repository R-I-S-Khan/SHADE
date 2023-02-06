import argparse
import subprocess
import os
from queue import Queue, Empty
from threading  import Thread

def enqueue_output(out, queue):
	for line in iter(out.readline, b''):
		queue.put(line.decode('utf-8'))
	out.close()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# parser.add_argument('script_to_run', type=str, help="Training code to be run")
	parser.add_argument('script_to_run', type=str, help="Training code to be run")
	parser.add_argument('train_paths', nargs="+", help="Path to the train set")
	parser.add_argument('-master_address', type=str,
						help='The address of master node to connect into')
	parser.add_argument('-master_socket', type=str,
						help='The NCCL master socket to connect into')
	parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
						help='number of nodes')
	parser.add_argument('-g', '--gpus', default=1, type=int,
						help='number of gpus per node')
	parser.add_argument('-nr', '--nr', default=0, type=int,
						help='ranking within the nodes')
	parser.add_argument('--epochs', default=1, type=int, metavar='N',
						help='number of total epochs to run')
	# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
	#                 metavar='W', help='weight decay (default: 1e-4)',
	#                 dest='weight_decay')
	# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
	#                 help='momentum')
	# parser.add_argument('--resume', default='', type=str, metavar='PATH',
	#                 help='path to latest checkpoint (default: none)')
	parser.add_argument('--batch_size', default=256, type=int, help="Batch size used for training and validation")
	parser.add_argument('--network', default='resnet18', type=str,
						help='''The neural network we use for the training. It currently
						supports alexnet, resnet50, vgg16 and resnet18. The script will instantiate resnet18 if
						any other argument is passed''')
	parser.add_argument('--master_port', default=8888, type=int,
						help='The port on which master is listening')
	parser.add_argument('--val_paths', nargs="+", default=None, help="Paths to the val set")
	parser.add_argument('--cache_training_data', default=False, action='store_true',
						help='Whether to cache the training data. Passing this option requires the system to have a redis in-memory cache installed on the localhost listening on port 6379')
	parser.add_argument('--nodedist', default=False, action='store_true',
						help='Set to run multi-node training.')
	parser.add_argument('-rep','--replication_factor', default=3.0, type=float, help='number of times data in cache will be repeated.')
	parser.add_argument('-wss','--working_set_size', default=0.1, type=float, help='percentage of dataset to be cached.')
	parser.add_argument('--host_ip', nargs='?', default='0.0.0.0', const='0.0.0.0', help='redis master node ip')
	parser.add_argument('--port_num', nargs='?', default=None, const='6379', help='port that redis cluster is listening to')
	parser.add_argument('-cn', '--cache_nodes', default=3, type=int,
						help='number of nodes cache is distributed across')


	args = parser.parse_args()

	print("****************** Hello, run begins *************************")

	subprocess.run('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches', shell=True)

	abs_train_paths = []
	for train_path in args.train_paths:
		abs_train_paths.append(os.path.abspath(train_path))

	cmd = ['python', os.path.abspath(args.script_to_run)] 
	cmd.extend(abs_train_paths)
	print(args.network)
	if args.nodedist:
		cmd.extend(['-master_address', args.master_address,\
				'-master_socket', args.master_socket,\
				'-n', str(args.nodes), '-g', str(args.gpus), '-nr', str(args.nr),
				 '--epochs', str(args.epochs), '--master_port', str(args.master_port), '--network', args.network, 
				 '--batch_size', str(args.batch_size), '--replication_factor', 
				 str(args.replication_factor), '-wss', str(args.working_set_size), 
				 '--host_ip', str(args.host_ip), '--port_num', str(args.port_num),
				 '-cn', str(args.cache_nodes)])
	if not args.nodedist:
		args.cache_nodes = 1
		cmd.extend(['-master_address', args.master_address,\
				'-master_socket', args.master_socket,\
				'-n', str(args.nodes), '-g', str(args.gpus), '-nr', str(args.nr),
				 '--epochs', str(args.epochs), '--master_port', str(args.master_port), '--network', args.network, 
				 '--batch_size', str(args.batch_size), '--replication_factor', 
				 str(args.replication_factor), '-wss', str(args.working_set_size),
				 '-cn', str(args.cache_nodes)])
	if args.cache_training_data:
		cmd.append('--cache_training_data')
	nvidia_smi_cmd = ["nvidia-smi", '-l', '1']
	sar_cmd = ["sar", "-u", "-r", "-b","-n", "DEV", "1"]
	print(args.val_paths)
	if args.val_paths is not None:
		for i in range(len(args.val_paths)):
			args.val_paths[i] = os.path.abspath(args.val_paths[i])
		cmd.append('--val_paths')
		cmd.extend(args.val_paths)
	training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	nvidia_process = subprocess.Popen(nvidia_smi_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	sar_process = subprocess.Popen(sar_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	train_log = open("train.log", 'w')
	nvidia_log = open("nvidia_smi_output_node%d.txt" % (args.nr + 1), 'w')
	sar_log = open("sar_output_node%d.txt" % (args.nr + 1), 'w')

	nvidia_q = Queue()
	nvidia_stdout_tracker = Thread(target=enqueue_output, args=(nvidia_process.stdout, nvidia_q))
	nvidia_stdout_tracker.daemon = True # thread dies with the program
	nvidia_stdout_tracker.start()


	sar_q = Queue()
	sar_stdout_tracker = Thread(target=enqueue_output, args=(sar_process.stdout, sar_q))
	sar_stdout_tracker.daemon = True # thread dies with the program
	sar_stdout_tracker.start()

	train_q = Queue()
	train_stdout_tracker = Thread(target=enqueue_output, args=(training_process.stdout, train_q))
	train_stdout_tracker.daemon = True # thread dies with the program
	train_stdout_tracker.start()

	training_process_termninated = False
	nvidia_has_data = True
	sar_has_data = True
	train_has_data = True
	while not(training_process_termninated) or nvidia_has_data or sar_has_data:
		#if not training_process_termninated:
		#    print("running")
		
		try:  line = nvidia_q.get_nowait() # or q.get(timeout=.1)
		except Empty:
			nvidia_has_data = False
		else:
			nvidia_has_data = True
			nvidia_log.write(line)


		try:  line = sar_q.get_nowait() # or q.get(timeout=.1)
		except Empty:
			sar_has_data = False
		else:
			sar_has_data = True
			sar_log.write(line)


		try:  line = train_q.get_nowait() # or q.get(timeout=.1)
		except Empty:
			train_has_data = False
		else:
			train_has_data = True
			train_log.write(line)

		if training_process.poll() is not None and training_process_termninated is False:
			training_process_termninated = True
			nvidia_process.terminate()
			sar_process.terminate()
			print("terminated")

	nvidia_log.close()
	train_log.close()
	sar_log.close()
	subprocess.run('sync; echo 3 | sudo tee /proc/sys/vm/drop_caches', shell=True) 
	print("****************** Farewell *************************")


