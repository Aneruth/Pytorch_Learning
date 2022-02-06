# Linear Regeression from scratch
import torch
import matplotlib.pyplot as plt
import os

torch.manual_seed(11)

X = torch.rand(10) # Defining X 
Y = torch.rand(10) # Defining Y
w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True) # Initialise weights 

# Caculating forward pass
def forward(x):
    return w * x

# Calculating the loss function
def loss(y,y_hat):
    # Since it is a linear regression we use MSE (Mean Square Error) as loss function
    return ((y_hat - y).pow(2)).mean()

# calculating the gradients
# MSE = 1/N * (w*x - y)**2
def gradients(x,y,y_hat):
    """A manual calculation for gradient

    Args:
        x (tensor): Input feature
        y (tensor): Target class
        y_hat (tensor): preciction

    Returns:
        torch: calculated gradient value Mean Sqaure error
    """
    return torch.dot(x.pow(2),y_hat-y).mean()

print(f'Prediction before training: {forward(5):.4f}')

lr = 0.1 # learning rate
epochs = 10
loss_val,acc_val = [],[]
for epoch in range(epochs):
    y_hat = forward(X) # predictions
    l = loss(Y,y_hat) # loss calculation
    # grad = gradients(X,Y,y_hat) # gradient calculation
    l.backward()
    # Updating the weights
    # w -= lr * grad # for manual calculation

    with torch.no_grad():
        w -= lr * w.grad

    # Zero gradients
    w.grad.zero_()
    if epoch % 1 == 0:
        loss_val.append(l.item())
        print(f'Epoch {epoch +1}: w = {w:.3f}, loss: {l:.8f}')
    
print(f'Prediction after training: {forward(5):.4f}')

# Visualising the loss function
plt.plot(loss_val, label="Training Loss")
plt.legend()
plt.title(f'model {"Loss Graph"}')
plt.ylabel("Loss Values")
plt.xlabel('Epochs')
plt.show(block=False)
plt.savefig(os.path.dirname(os.path.realpath(__file__))+'/LossGraph.png')
plt.pause(3)
plt.close()