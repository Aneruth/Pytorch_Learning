import torch,math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

w = torch.tensor(1.0,requires_grad=True)

# Forward pass to compute the loss
y_hat = w * x

# If our values are scalar use .pow(2).sum() else .pow(2)
loss = (y_hat - y).pow(2).sum()

print(loss)

# backward pass
loss.backward() # Gradient Computaion
print(w.grad)

# Updating the weights
# Compute forwards and backward for couple of iterations
# A simple back propogation
epoch = 100
for epoch in range(epoch):
    y_hat = w * x
    loss = (y_hat - y).pow(2).sum()
    if epoch % 2 == 0:
        print(epoch,loss.item())
    grad_y_pred = 2.0 * (y_hat - y)