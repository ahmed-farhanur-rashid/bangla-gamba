import torch
import torch.nn as nn
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

device = "cuda"
hidden = torch.randn(2, 4, 16, device=device, requires_grad=True)
weight = nn.Parameter(torch.randn(100, 16, device=device))
targets = torch.randint(0, 100, (2, 4), device=device)

criterion = LigerFusedLinearCrossEntropyLoss(lse_square_scale=1e-4, return_z_loss=False)
output = criterion(weight, hidden.view(-1, 16), targets.view(-1))
print("Loss type:", type(output))
print(output)
