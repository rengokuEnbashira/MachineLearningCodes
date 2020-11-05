import numpy as np
import torch
import torch.nn as nn
import  sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import sys 

# Creating Class for Multilayer Perceptron (1 input layer, 1 hidden layer, 1 output layer)
class MultilayerPerceptron(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MultilayerPerceptron,self).__init__() # Calling super class constructor
        self.fc1 = nn.Linear(input_size,hidden_size) # Input Layer <-> Hidden Layer (Linear Trasnformation: input*weights)
        self.relu = nn.ReLU() # Input Layer <-> Hidden Layer (Activation Function - give non linearity to the previous transformation)
        self.fc2 = nn.Linear(hidden_size,output_size) # Hidden Layer <-> Output Layer (Linear Transformation)
    def forward(self,x):
        o = self.fc1(x) # eval first linear transformation to the input
        o = self.relu(o) # apply activation function to the previous result
        o = self.fc2(o) # eval second linear transformation to the previous result
        return o

def train(model,criterion,optimizer,train_loader): # training
    model.train() # Start training 
    acc_loss = 0 # accumulate loss
    for batch_idx,(images,labels) in enumerate(train_loader):
        optimizer.zero_grad() # initialize the gradient
        o = model(images) # eval the model with the current input

        loss = criterion(o,labels) # calculate the loss 
        acc_loss += loss # add the loss to the accumulate loss

        loss.backward() # backpropagation
        optimizer.step() # update the parameters (weights, bias) of the model

    train_loss = acc_loss/len(train_loader) # calculate the average loss of these training step
    return train_loss

data = ds.load_digits() # load numbers dataset 
x,y = data.images, data.target 

if sys.argv[1] == "mlp":
    input_size = 64
    hidden_size = 20
    num_classes = 10
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.01

    x = x.reshape(1797,64)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    x_train_tensor = torch.tensor(x_train).float()
    y_train_tensor = torch.tensor(y_train).long()
    
    x_test_tensor = torch.tensor(x_test).float()
    y_test_tensor = torch.tensor(y_test).long()
    
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor,y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    
    model = MultilayerPerceptron(input_size,hidden_size,num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        loss = train(model,criterion,optimizer,train_loader)
        print(epoch,loss)
    
    print(model(x_test_tensor))
