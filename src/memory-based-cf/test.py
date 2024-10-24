import sys
print(sys.executable)

import torch
print(torch.cuda.is_available())  # 如果返回 True，表示 CUDA 可用
print(torch.version.cuda)