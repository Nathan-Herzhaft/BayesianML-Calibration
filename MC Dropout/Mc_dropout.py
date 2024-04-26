#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import pi
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim

from Architectures import Network_16, Network_8, Network_4, Network_2

#%%
class CreateDataset(Dataset) :
    def __init__(self,X,y) :
        self.X = X
        self.y = y

    def __len__(self) :
        return self.X.size
    
    def __getitem__(self,index) :
        input = self.X[index]
        target = self.y[index]
        return torch.Tensor(input[...,None]), torch.Tensor(target[...,None])
    

def true_model(x) :
    return np.sin(2*pi*x) +  2
        

def generate_loaders(true_model=true_model,noise=0.1,n_samples=1000) :

    X = np.random.uniform(-1,1,n_samples)
    y = true_model(X) + np.random.normal(0,noise,n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(X_train,y_train,color='b',label ='train')
    ax.scatter(X_test,y_test,color='r',label='test')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()

    train_data = CreateDataset(X_train,y_train)
    test_data = CreateDataset(X_test,y_test)
    loaders = {
    'train' : DataLoader(train_data,
                         batch_size = 100,
                         shuffle=True),

    'test' : DataLoader(test_data,
                        batch_size=100)
    }

    return loaders

loaders = generate_loaders()




# %%
def train(model,num_epochs,train_loader, verbose) :

    model.train()
    
    print('Training ', model.__name__)
    for epoch in range(num_epochs) :
        for i, (input, target) in enumerate(train_loader) :
            output = model(input)
            loss = loss_func(output,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % verbose == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}' .format(epoch + 1, num_epochs, loss.item()))


def plot_test(model,test_loader) :
    list_output = []
    list_target = []
    list_input = []
    model.eval()
    with torch.no_grad() :
        for (input, target) in test_loader :
            output = model(input).detach().numpy()
            list_output.append(output)
            list_target.append(target.detach().numpy())
            list_input.append(input.detach().numpy())

    targets = list_target[0]
    outputs = list_output[0]
    inputs = list_input[0]
    for output in list_output[1:] :
        outputs = np.concatenate([outputs,output])
    for target in list_target[1:] :
        targets = np.concatenate([targets,target])
    for input in list_input[1:] :
        inputs = np.concatenate([inputs,input])
    plt.scatter(inputs,outputs,color='red',label='Outputs')
    plt.scatter(inputs,targets,color='black', label='Targets')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('y')




#%%
model = Network_2(512,0.2)

loss_func  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr= 0.01)

train(model,20,loaders['train'],5)
plot_test(model,loaders['test'])



# %%
def enable_dropout(model) :
    for m in model.modules() :
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def mc_dropout(model, test_loader, n_samples) :
    sample_size = test_loader.dataset.__len__()
    Outputs = np.zeros((sample_size,n_samples))

    Inputs = test_loader.dataset.X
    index = np.argsort(Inputs)
    Inputs = Inputs[index]
    Targets = test_loader.dataset.y[index]

    model.eval()
    enable_dropout(model)

    with torch.no_grad() :
        for k in range(n_samples) :
            i = 0
            for (input, target) in test_loader :
                outs = model(input).detach().numpy()
                Outputs[i:i+outs.size,k:k+1] = outs
                i += outs.size
            Outputs[:,k:k+1] = Outputs[:,k].T[index][...,None]

    Means = Outputs.mean(axis=1).T
    Deviations = Outputs.std(axis=1).T


    return Inputs, Targets, Outputs.T, Means, Deviations


def plot_mc_dropout(Inputs,Targets,Means,Deviations,true_model) :
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(Inputs,Targets,color='brown',s=0.5,label='observations')
    ax.plot(Inputs,true_model(Inputs),color='black',label='true model')
    ax.plot(Inputs,Means,color='blue',label='posterior mean')
    ax.fill_between(Inputs,Means-Deviations,Means+Deviations,color = 'b', alpha = 0.1, label='posterior deviation')
    ax.legend(loc='upper right')
    ax.set_xlabel('X')
    ax.set_ylabel('y')


def plot_mc_dropout_samples(Inputs,Targets,Outputs,true_model,n_samples) :
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(Inputs,Targets,color='brown',s=0.7,label='observations')
    ax.plot(Inputs,true_model(Inputs),color='black',label='true model')
    for k in range(n_samples) :
        if k==0:
            
            ax.plot(Inputs,Outputs[k], linewidth=0.5,label='samples')
        else :
            ax.plot(Inputs,Outputs[k], linewidth=0.5)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('y')


def plot_mc_dropout_interval(model,x_min,x_max,n_samples,test_loader) :
    x = torch.Tensor(np.linspace(x_min,x_max)[...,None])

    Inputs = test_loader.dataset.X
    Targets = test_loader.dataset.y

    Outputs = np.zeros((x.__len__(),n_samples))

    model.eval()
    enable_dropout(model)

    
    for k in range(n_samples) :
        y = model(x).detach().numpy()
        Outputs[:,k:k+1] = y

    Means = Outputs.mean(axis=1).T
    Deviations = Outputs.std(axis=1).T
    x = x.detach().numpy().T[0]
    print(x)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(Inputs,Targets,color='red',s=0.5,label='observations')
    ax.plot(x,Means,color='blue',label='posterior mean')
    ax.fill_between(x,Means-Deviations,Means+Deviations,color = 'b', alpha = 0.1, label='posterior deviation')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('y')


#%%

Inputs, Targets, Outputs, Means, Deviations = mc_dropout(model, loaders['test'], 10)

plot_mc_dropout_samples(Inputs,Targets,Outputs,true_model,3)
plot_mc_dropout(Inputs,Targets,Means,Deviations,true_model)
plot_mc_dropout_interval(model,-1,1,10,loaders['test'])


# %%
