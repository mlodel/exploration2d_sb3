import torch
import torch.nn as nn
import time
from torch.nn import functional as F


print("PyTorch version: ", torch.__version__)
print("CUDA version: ", torch.version.cuda)
print("CUDNN version: ", torch.backends.cudnn.version())
print("GPU name: ", torch.cuda.get_device_name(0))
print("GPU is available: ", torch.cuda.is_available())
print("GPU count: ", torch.cuda.device_count())
print("GPU current device: ", torch.cuda.current_device())

# Test CUDA
def profile(model, x, benchmark, nb_iters):
    torch.backends.cudnn.benchmark = benchmark

    # warmup
    for _ in range(10):
        out = model(x)

    shape = out.shape

    y = torch.rand(shape, device=x.device)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(nb_iters):
        out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()
    torch.cuda.synchronize()
    t1 = time.time()

    return (t1 - t0) / nb_iters


model1 = nn.Sequential(
    nn.Conv1d(24, 256, kernel_size=(12,), stride=(6,), groups=4),
    nn.ReLU(),
    nn.Conv1d(256, 256, kernel_size=(6,), stride=(3,), padding=(2,), groups=4),
    nn.ReLU(),
    nn.Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), groups=4),
    nn.ReLU(),
).cuda()

x = torch.randn(64, 24, 224, device='cuda')
print(torch.cuda.get_device_name(0))
time0 = profile(model1, x, benchmark=False, nb_iters=100)
print('Conv1d model, benchmark=False, {:.3f}ms/iter'.format(time0*1000))
time1 = profile(model1, x, benchmark=True, nb_iters=100)
print('Conv1d model, benchmark=True, {:.3f}ms/iter'.format(time1*1000))

model2 = nn.Sequential(
    nn.Conv2d(8, 32, kernel_size=(8, 8), stride=(4, 4)),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU()
).cuda()

x = torch.randn(64, 8, 224, 224, device='cuda')
time0 = profile(model2, x, benchmark=False, nb_iters=100)
print('Conv2d model, benchmark=False, {:.3f}ms/iter'.format(time0*1000))
time1 = profile(model2, x, benchmark=True, nb_iters=100)
print('Conv2d model, benchmark=True, {:.3f}ms/iter'.format(time1*1000))
