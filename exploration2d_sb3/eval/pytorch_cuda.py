import torch
import time

print("PyTorch version: ", torch.__version__)
print("CUDA version: ", torch.version.cuda)
print("CUDNN version: ", torch.backends.cudnn.version())
print("GPU name: ", torch.cuda.get_device_name(0))
print("GPU is available: ", torch.cuda.is_available())
print("GPU count: ", torch.cuda.device_count())
print("GPU current device: ", torch.cuda.current_device())

# Test CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start = time.time()
x = torch.rand(3, 3).to(device)
print("Time to create tensor: ", time.time() - start)

start = time.time()
y = torch.rand(1000, 1000).to(device)
print("Time to create tensor: ", time.time() - start)
