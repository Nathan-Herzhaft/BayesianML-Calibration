#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import pi
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
from torchhk import transform_model
import torchbnn as bnn
from Architectures import Network_1, Network_2, Network_4, Network_8, Network_16


#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# %%
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
def set_bayesian(model) :
    transform_model(model, nn.Linear, bnn.BayesLinear,
                args = {"prior_mu":0, "prior_sigma":0.1, "in_features" : ".in_features",
                  "out_features" : ".out_features", "bias":".bias"
                 },
                 attrs={"weight_mu" : ".weight"})
    return model
# %%
