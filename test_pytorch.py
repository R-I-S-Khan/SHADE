import torch
import torchvision
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.nccl.version())
#conda list -n metis_conda -f pytorch