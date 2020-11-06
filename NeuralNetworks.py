import numpy as np
import torch
import torch.nn as nn
import  sklearn.datasets as ds
from sklearn.model_selection import train_test_split
import sys 

# Creating Class for Multilayer Perceptron with 3 layers
# (1 input layer, 1 hidden layer, 1 output layer)
class MultilayerPerceptron(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(MultilayerPerceptron,self).__init__()      # Calling super class constructor
        self.fc1 = nn.Linear(input_size,hidden_size)     # Input Layer <-> Hidden Layer (Linear Trasnformation: input*weights)
        self.relu = nn.ReLU()                            # Input Layer <-> Hidden Layer (Activation Function - give non linearity to the previous transformation)
        self.fc2 = nn.Linear(hidden_size,output_size)    # Hidden Layer <-> Output Layer (Linear Transformation)
    def forward(self,x):
        o = self.fc1(x)                                  # eval first linear transformation to the input
        o = self.relu(o)                                 # apply activation function to the previous result
        o = self.fc2(o)                                  # eval second linear transformation to the previous result
        return o

class ConvNN(nn.Module):
    def __init__(self,):
        super(ConvNN,self).__init__()                # Calling super class (nn.Module) constructor
        self.layer1 = nn.Sequential(                 # Create a sequential model for first layer
            nn.Conv2d(1,10,kernel_size=3,padding=0), # Convolutional layer: 1 x 8 x 8 (x) 10 x 3 x 3 -> 10 x 6 x 6
            nn.BatchNorm2d(10),                      # Batch normalization (s relu does not blow up)
            nn.ReLU(),                               # Activation Function: ReLU
            nn.MaxPool2d((2))                        # Max Pooling: 10 x 6 x 6 -> 10 x 3 x 3
        )
        self.layer2 = nn.Sequential(                 # Second Layer
            nn.Conv2d(10,20,kernel_size=3,padding=0),# Convolution Layer: 10 x 3 x 3 (x) 2 x 3 x 3 -> 20 x 1 x 1
            nn.ReLU(),                               # Activation Function: ReLU
        )
        self.fc = nn.Linear(20,10)                   # Fully connected layer
    def forward(self,x):
        o = self.layer1(x)
        o = self.layer2(o)
        o = o.view(o.size(0),-1)
        o = self.fc(o)
        return o

def train(model,criterion,optimizer,train_loader): # training
    model.train() # Start training 
    acc_loss = 0  # accumulate loss
    for batch_idx,(images,labels) in enumerate(train_loader):
        optimizer.zero_grad() # initialize the gradient
        o = model(images)     # eval the model with the current input

        loss = criterion(o,labels) # calculate the loss 
        acc_loss += loss # add the loss to the accumulate loss

        loss.backward()  # backpropagation
        optimizer.step() # update the parameters (weights, bias) of the model

    train_loss = acc_loss/len(train_loader) # calculate the average loss of these training step
    return train_loss

data = ds.load_digits()        # load numbers dataset 
x,y = data.images, data.target # x.shape = (1797,8,8) , y.shape = (1797)

if sys.argv[1] == "mlp":
    input_size = 64
    hidden_size = 20
    num_classes = 10
    num_epochs = 30
    batch_size = 100
    learning_rate = 0.01

    x = x.reshape(1797,64)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    x_train_tensor = torch.tensor(x_train).float() # converting x train data to tensor
    y_train_tensor = torch.tensor(y_train).long()  # converting y train data to tensor
    
    x_test_tensor = torch.tensor(x_test).float()   # converting x test data to tensor
    y_test_tensor = torch.tensor(y_test).long()    # converting y test data to tensor
    
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor) # creating train dataset using x train and y train tensors
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor,y_test_tensor)    # creating test dataset using x test and y test tensors

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size) # creating train data loader using train dataset
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)   # create test data loader using test dataset
    
    model = MultilayerPerceptron(input_size,hidden_size,num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        loss = train(model,criterion,optimizer,train_loader)
        print(epoch,loss)
    pred = model(x_test_tensor[:30])
    sm = nn.Softmax(dim=1)
    pred = sm(pred)
    print(pred)
    print(pred.max(1))
    print(y_test_tensor[:30])

elif sys.argv[1] == "cnn":
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.01

    x = x.reshape(1797,1,8,8)
    x_train, x_test, y_train, y_test = train_test_split(x,y)

    x_train_tensor = torch.tensor(x_train).float() # converting x train data to tensor
    y_train_tensor = torch.tensor(y_train).long()  # converting y train data to tensor
    
    x_test_tensor = torch.tensor(x_test).float()   # converting x test data to tensor
    y_test_tensor = torch.tensor(y_test).long()    # converting y test data to tensor
    
    train_dataset = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor) # creating train dataset using x train and y train tensors
    test_dataset = torch.utils.data.TensorDataset(x_test_tensor,y_test_tensor)    # creating test dataset using x test and y test tensors

    train_loader = torch.utils.data.DataLoader(train_dataset) # creating train data loader using train dataset
    test_loader = torch.utils.data.DataLoader(test_dataset)   # create test data loader using test dataset
    model = ConvNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(num_epochs):
        loss = train(model,criterion,optimizer,train_loader)
        print(epoch,loss)
    _,pred = model(x_test_tensor[:30]).max(1)
    print(pred)
    print(y_test_tensor[:30])
