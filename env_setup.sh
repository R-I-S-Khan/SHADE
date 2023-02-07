#!/bin/bash
SHADE_HOME=$(pwd)
echo export SHADE_HOME=$(pwd) >> ~/.bashrc


installConda() {
  conda list &> /dev/null
  if [ $? -eq 0 ]; then
    echo "Anaconda environment already installed"
    echo "Using existing conda environment."
    #conda update conda
    #conda install anaconda=2022.10
  else
    conda_dir=$HOME/anaconda3
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
    bash Anaconda3-2022.10-Linux-x86_64.sh -b -p $conda_dir
    export PATH=$conda_dir/bin:$PATH
  fi
}

installConda conda

# create SHADE conda env
conda init bash
. ~/.bashrc
#conda env create -f shade_conda.yaml # install shade_conda dependencies
yes Y | conda create --name shade_conda python=3.8
conda activate shade_conda
pip install redis-py-cluster
yes Y | conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
yes Y | conda install heapdict
yes Y | conda install pandas
#yes Y | conda install -c pytorch magma-cuda112 ## Add LAPACK support for the GPU if needed

if [ "$1" == "--redis" ]; then
  cd $HOME
  wget -P $HOME/ http://download.redis.io/releases/redis-6.0.1.tar.gz 
  tar xzf redis-6.0.1.tar.gz
  mv $HOME/redis-6.0.1 $HOME/redis-stable
  cd $HOME/redis-stable
  yes Y | sudo apt-get install tcl
  make
fi

#setting sys parameters for redis
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo sysctl vm.overcommit_memory=1
echo never | sudo tee -a /sys/kernel/mm/transparent_hugepage/enabled
echo 1024 | sudo tee -a /proc/sys/net/core/somaxconn
echo 'net.core.somaxconn=65535' | sudo tee -a /etc/sysctl.conf
#clearing sys cache
croncmd="sync && echo 3 | sudo tee /proc/sys/vm/drop_caches"
cronjob="* * * * * $croncmd"
( crontab -l | grep -v -F "$croncmd" || : ; echo "$cronjob" ) | crontab -
#sar setup
sudo apt-get update
yes Y | sudo apt-get install sysstat
sudo sed -i 's/ENABLED=.*/ENABLED="true"/' /etc/default/sysstat

#installing pytorch
cd $HOME
git clone --recursive https://github.com/pytorch/pytorch
cd $HOME/pytorch
git checkout tags/v1.7.0
#git clean -xdf
#python setup.py clean
git submodule sync
#git submodule deinit -f .
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
cp -r $HOME/anaconda3/envs/shade_conda/lib $HOME/lib #saving lib in case it is needed later.
yes Y | conda install torchvision=0.8.1 -c pytorch
python setup.py install
#cp -r $HOME/lib $HOME/anaconda3/envs/shade_conda/ #this might be needed to overwrite the libs created by torchvision
#conda remove pandas #pandas might create issues, hence remove the earlier version
#pip install pandas #do a reinstallation
#pip3 install --upgrade numpy #upgrade numpy
#might need to add /home/cc/anaconda3/envs/shade_conda/lib in LD_LIBRARY_PATH.


