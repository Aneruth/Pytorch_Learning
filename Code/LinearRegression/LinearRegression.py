import torch,warnings,os,matplotlib.pyplot as plt
import torch.nn as nn

warnings.simplefilter("ignore", UserWarning)

torch.manual_seed(2)

X = torch.rand((10000,3)) # Defining X 
y = torch.rand((10000,1)) # Defining Y

n_samples,n_features = X.shape

inp_size = n_features
out_size = n_samples

class LinearRegression(nn.Module):
    def __init__(self,inputDim,outputDim):
        super(LinearRegression,self).__init__()

        # Layers
        self.layer1 = nn.Linear(inputDim, outputDim)
        self.layer2 = nn.Linear(inputDim, outputDim)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(x)
        return out

model = LinearRegression(inp_size, out_size)

lr = 0.1 # learning rate
epochs = 20

loss = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(),lr=lr)

def train(y,loss_val = []):
    """Train loop for our model

    Args:
        y (Tensor): Prediction labels
        loss_val (list, optional): A list to append all the loss values. Defaults to []

    Returns:
        list: loss values for plotting
    """
    for epoch in range(epochs):
        y_hat = model(X) # predictions
        l = loss(y,y_hat) # loss calculation
        # grad = gradients(X,Y,y_hat) # gradient calculation
        l.backward()

        # Upadting the weights
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad()

        print ("epoch #",epoch)
        print ("loss: ", l.item())
        loss_val.append(l.item())
    return loss_val

def plotGraph( data):
    """A funtion to plot the graph

    Args:
        data (list): An input list to plot the graph.
    """
    plt.plot(data, label="Training Loss")
    plt.legend()
    plt.title(f'model {"Loss Graph"}')
    plt.ylabel("Loss Values")
    plt.xlabel('Epochs')
    plt.show(block=False)
    plt.savefig(os.path.dirname(os.path.realpath(__file__))+'/LossGraph.png')
    plt.pause(3)
    plt.close()

plotGraph(train(y))