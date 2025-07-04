import os
import torch

print(os.environ['CONDA_DEFAULT_ENV'])
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print("test")