#!/bin/bash
#transfer shade utils
cp shade_utils/shadesampler.py $HOME/pytorch/torch/utils/data/
cp shade_utils/shadedataset.py $HOME/pytorch/torch/utils/data/
cp shade_utils/__init__.py $HOME/pytorch/torch/utils/data/
cd $HOME/pytorch
python setup.py install