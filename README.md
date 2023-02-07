# SHADE

**SHADE is a scalable open-source Deep Learning Training(DLT)-aware caching system.**

SHADE detects fine-grained importance variations at per-sample level and leverages the variance to
make informed caching decisions for a distributed DLT job. It is built on top of PyTorch and has high-level APIs to train on DL models and datasets.

## Getting Started

### Installation (Linux)

You can run `env_setup.sh` in all of the client and storage nodes. If you don't have CUDA available you can install it from [NVIDIA](https://developer.nvidia.com/cuda-toolkit) before running the setup script.

```
source env_setup.sh # Add `--redis` if you don't have a redis installation available. 
```

Update `env_setup.sh` if you prefer different versions of conda or Redis.
Once the `env_setup.sh` has finished execution, check it.

```
python test_pytorch.py  
```
If the installation of PyTorch and Torchvision was successful you would not see any errors. If you find errors, you can try some of the extra steps mentioned in the `env_setup.sh` (e.g., fixing LD_LIBRARY_PATH, upgrading packages) If you are still getting errors, you might need to completely install a new version of Conda, create a new Python environment, and then re-install PyTorch.

If the PyTorch and Torchvision installation is successful, then invoke the 'transfer_utils.sh' script. This will integrate SHADE utils with the PyTorch source code and install it.

```
bash transfer_utils.sh  
```

Check whether the installation has completed successfully by invoking the test_pytorch.py script.

```
python test_pytorch.py  
```

### Running SHADE

Before running SHADE, you need to first setup your storage and client nodes. Make sure the relevant ports are open and all of the nodes recognize each other by adding IP information in the `/etc/hosts` file. If you plan to use a remote storage, then you would need to install a filesystem, e.g. NFS server and mount the remote storage to all of your client nodes. Check if you can access all nodes from one another using ssh. You can also use a local storage if you decide to run single-node training.

To setup the cache, you would first need to change the `redis.conf` file. Some relevant parameters that you might want to change include the bind point, port, maxmemory, client-query-buffer-limit, io-threads, persistence, etc. If you want to train on a single node, you would need to comment out the cluster parameters. Otherwise, keep the cluster parameters. Then you need to deploy the Redis servers on all of the client nodes. 

```
screen -S redis-server $HOME/redis-stable/src/redis-server $HOME/redis-stable/redis.conf
```

If you are running multi-node training, you should then create a cluster.
```
$HOME/redis-stable/src/redis-cli --cluster create NODE1_IP:PORT NODE2_IP:PORT NODE3_IP:PORT --cluster-replicas 0
```

#### Multi-node Training
Once you are done setting up Redis, placed the data you want to train on the remote storage, and made sure all the nodes are accessible from one another, you are ready to run SHADE. We have prepared a script so that you can start by benchmarking SHADE on CIFAR-10 dataset.

At first, prepare the CIFAR-10 dataset in [Imagefolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) format by running the following command. It will save the CIFAR-10 dataset at $HOME/datasets/.

```
python prepare_CIFAR10.py -train_src cifar10_data/ -test_src cifar10_data/ -train_dest $HOME/datasets/cifar10/train/ -test_dest $HOME/datasets/cifar10/test/
```

Once you have prepared the dataset, you can now run shade using SHADE's analysis framework. Assume, the CIFAR-10 dataset is located on a remote storage which is accessible using the mounting point /mnt/nfs/cifar10. The master node IP and socket are 10.X.X.X and eno1. Now let's run SHADE on four nodes each having 2 GPUs.

```
cd benchmarks
```
Now you will need to run the following command in all of your client nodes. In the terminal of each client node the command will be same, the only difference being the rank number (nr). For first, second, third, and fourth node the value of nr will be 0,1,2, and 3 respectively. `-rep` means the number of times samples in cache will be replicated, `-wss` means the percentage of dataset you will cache, and `-cn` means the number of Redis nodes that you are using as a distributed cache. In this case, we are caching 20% of CIFAR-10 dataset (~10000 samples), replication factor is 3, and the number of Redis nodes in the distributed cache is 4 (as we have created a redis cluster with all of the four client nodes.) 
```
screen -S train python runshade.py bench_cifar10.py /mnt/nfs/cifar10/train/ -master_address 10.X.X.X -master_socket eno1 -n 4 -g 2 -nr 0 --epochs 5 --network resnet18 --batch_size 64 --val_paths /mnt/nfs/cifar10/test/ --cache_training_data --nodedist -rep 3.0 -wss 0.2 --host_ip [Redis_host_ip] --port_num [redis_port] -cn 4
```
Detach the screen session using `Ctrl+A+D`

#### Single-node Training

If you want to train on single node, you will need to run Redis in standalone mode by editing the redis.conf file. Start the redis instance.

```
screen -S redis-server $HOME/redis-stable/src/redis-server $HOME/redis-stable/redis.conf
```
Next, run the following command after editing the necessary parameters.
```
screen -S train python runshade.py bench_cifar10.py /mnt/nfs/cifar10/train/ -master_address 10.X.X.X -master_socket eno1 -n 1 -g 2 -nr 0 --epochs 5 --network resnet18 --batch_size 64 --val_paths /mnt/nfs/cifar10/test/ --cache_training_data -rep 2.0 -wss 0.2 -cn 1
```
Detach the screen session using `Ctrl+A+D`

Once the training ends, you can take the look at the generated log files.

Let's look at the number of hits(H) and misses(M).

```
cat train.log | grep hitting | wc -l
cat train.log | grep miss | wc -l
```
You can then find out the hit rate by dividing H with the sum of H and M.

Feel free to play around with different models, datasets, replication factor, working set sizes, etc. You might discover something interesting!

## Reference
Please read and/or cite as appropriate to use SHADE.

```bibtex
@inproceedings {285774,
author = {Redwan Ibne Seraj Khan and Ahmad Hossein Yazdani and Yuqi Fu and Arnab K. Paul and Bo Ji and Xun Jian and Yue Cheng and Ali R. Butt},
title = {{SHADE}: Enable Fundamental Cacheability for Distributed Deep Learning Training},
booktitle = {21st USENIX Conference on File and Storage Technologies (FAST 23)},
year = {2023},
isbn = {978-1-939133-32-8},
address = {Santa Clara, CA},
pages = {135--152},
url = {https://www.usenix.org/conference/fast23/presentation/khan},
publisher = {USENIX Association},
month = feb,
}
```









