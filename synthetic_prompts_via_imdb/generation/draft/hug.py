import torch

while True:
    z = torch.randn(1, 1024, device="cuda:4")
    