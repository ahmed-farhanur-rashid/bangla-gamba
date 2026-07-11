import torch
from liger_kernel.transformers import LigerRMSNorm

x = torch.randn(2, 4, 8, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
norm = LigerRMSNorm(16).cuda().bfloat16()
y = norm(x)
loss = y.sum()
loss.backward()
print("Success:", y.shape)
