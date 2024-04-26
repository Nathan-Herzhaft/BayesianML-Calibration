#%%
import sklearn
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import optim

from sklearn.datasets import make_blobs, make_circles, make_moons, make_classification



# %%
center_1 = np.array([[0,0]])
center_2 = np.array([[3,3]])

def plot_data(X,y) :
    fig,ax = plt.subplots(figsize=(6,6))
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.scatter(X[:,0],X[:,1],c=y,cmap='bwr')
    plt.show()

def make_data_blobs(center_1,center_2, stds, n_samples, plot=True) :
    centers= np.concatenate([center_1,center_2])
    X,y = make_blobs(n_samples=n_samples, n_features=2, centers=centers, cluster_std=stds)
    if plot :
        plot_data(X,y)
    return X,y[...,None]

def make_data_moons(n_samples,plot=True) :
    X,y = make_moons(n_samples=n_samples)
    if plot :
        plot_data(X,y)
    return X,y[...,None]

X,y = make_data_moons(100)


# %%
class CreateDataset(Dataset) :
    def __init__(self,X,y) :
        self.X = X
        self.y = y

    def __len__(self) :
        return self.X.shape[0]
    
    def __getitem__(self,index) :
        input = self.X[index]
        target = self.y[index]
        return torch.Tensor(input), torch.Tensor(target)




def make_loaders(X,y) :
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
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

loaders = make_loaders(X,y)




# %%

class Network_4(nn.Module) :
    def __init__(self, hidden_size, p) :
        super(Network_4,self).__init__()

        self.name = 'Network_4'

        self.Layer1 = nn.Sequential(
            nn.Linear(
            in_features=2,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )
        
        self.dropout1 = nn.Dropout(p)

        self.Layer2 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout2 = nn.Dropout(p)

        self.Layer3 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout3 = nn.Dropout(p)

        self.Layer4 = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = hidden_size,
            bias = True),

            nn.ReLU()
        )

        self.dropout4 = nn.Dropout(p)


        self.output = nn.Sequential(
            nn.Linear(
            in_features=hidden_size,
            out_features = 1,
            bias=True),

            nn.Sigmoid()
        )
        

    def forward(self,x) :
        x = self.Layer1(x)
        x = self.dropout1(x)
        x = self.Layer2(x)
        x = self.dropout2(x)
        x = self.Layer3(x)
        x = self.dropout3(x)
        x = self.Layer4(x)
        x = self.dropout4(x)
        out = self.output(x)
        return out
    



def train(model,num_epochs,train_loader,verbose) :
    
    optimizer = optim.Adam(model.parameters(), lr= 0.01)
    loss_func  = nn.BCELoss()
    
    model.train()
    
    print('Training ', model.name)
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

    model.eval()
    X = test_loader.dataset.X

    Outputs = []

    with torch.no_grad() :
        for (input, target) in test_loader :
            output = model(input).detach().numpy()
            Outputs.append(output)
    
    Outputs = np.concatenate(Outputs)

    plot_data(X,Outputs)


def enable_dropout(model) :
    for m in model.modules() :
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


# %%
model = Network_4(64,0.2)
train(model,100,loaders['train'],verbose=10)
plot_test(model,loaders['test'])


# %%

def plot_areas(model,x1_lims,x2_lims,n, n_samples,X_data,y_data) :
    model.eval()
    enable_dropout(model)

    X_1 = np.linspace(x1_lims[0],x1_lims[1],n)
    X_2 = np.linspace(x2_lims[0],x2_lims[1],n)
    X = np.array([[[X_1[i],X_2[j]] for i in range(n)] for j in range(n)])


    Z = np.zeros([n,n])
    for sample in range(n_samples) :
        for i in range(n) :
            for j in range(n) :
                with torch.no_grad() :
                    Z[i,j] += model(torch.tensor([X[i,j]],dtype=torch.float32)).detach().numpy()[0,0]

    Z = Z/n_samples


    fig, ax = plt.subplots(figsize=(6,6))
    ax.contourf(X_1,X_2,Z,cmap = 'coolwarm')
    ax.scatter(X_data[:,0],X_data[:,1],c=y_data,cmap='bwr')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_xlim(x1_lims[0],x1_lims[1])
    ax.set_ylim(x2_lims[0],x2_lims[1])

    return Z
    

Z = plot_areas(model,[-1,2],[-1,2],100,10,X,y)
# %%
