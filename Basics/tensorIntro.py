import torch

if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

x = torch.empty(3) # 1D tensor
x2d = torch.empty(2,3) # 2D tensor
x3d = torch.empty(2,3,3) # 3D tensor
print("\n", x, "\n", x2d, "\n", x3d)

# Tensor with all ones with a specific dimension
x1s = torch.ones(1,4)
print(f"\n{x1s}")

# By default the tensors type is float32 to change it we can use dtype={required type}
xInt = torch.ones(1,4,dtype=int)
print(f"\n{xInt}")

# To check the size of the tensors
print(f"\n{xInt.size()}")

# To create a new tensor from a list
aList = [i for i in range(xInt.size()[1])]
listTorch = torch.tensor(aList)
print(f"\n{listTorch}")

# Create a two random value of tensors
xRand = torch.rand(5,3)
yRand = torch.rand(5,3)

# Addition of two tensors
print(f"\n{torch.add(xRand,yRand)}")

# Inplace addition of two random values
print(f"\n{yRand.add_(xRand)}") # This will modify our torch y

# Just like pandas in tensors to fetch only one row or column we can slice it
# [row,col]
print(f"\nRandom values for XRand:\n {xRand}")
print(f"column Tensor {xRand[:,1]}")

# To fetch only the values we use .item()
print(f"\nSingle value for XRand at index [0,1] is {xRand[0,1].item()}")

# Reshaping the tensors
print(f"\nReshaped tensor is {torch.reshape(xRand,(3,5))}")

# convert from numpy to tensor or viceversa
import numpy as np
a = torch.ones(2)
b = a.numpy()
print(f"\nConverting from tensor to numpy {b} and the type is {type(b)}")

# Convert numpy to tensors
a = np.ones(5)
print(f"\nTensor data from numpy array is {torch.from_numpy(a)}")