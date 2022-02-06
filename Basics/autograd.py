import torch

# To calculate the gradients of the tensors
x = torch.randn(12,requires_grad=True)
print(x)