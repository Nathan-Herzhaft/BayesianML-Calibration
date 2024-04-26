#%%
import numpy as np
import matplotlib.pyplot as plt
import math



#%%
####################################################################
#               Fonctions de base
####################################################################

def Gaussian1D(mean,std) :
    def f(x) :
        Z = (2*math.pi*(std**2))**(1/2)
        exponent = - ((x - mean)**2) / ((2*(std**2)))
        return (1/Z)*np.exp(exponent)
    return f


def Gaussian2D(µ,S) :
    def f(x) :
        Z = (2*math.pi)*((np.linalg.det(S))**(1/2))
        exponent = (-1/2) * (x - µ)@np.linalg.inv(S)@((x - µ)[...,None])
        return ((1/Z) * np.exp(exponent))[0]
    
    return f



#%%
####################################################################
#           Initialisation du prior et des données
####################################################################

def generate_data(µ_1,µ_2,S_1,S_2,sample_size_1,sample_size_2,plot=True) :
    Class_1 = np.random.multivariate_normal(µ_1,S_1,sample_size_1)
    Class_2 = np.random.multivariate_normal(µ_2,S_2,sample_size_2)
    
    t = np.array([1 for x in Class_1] + [0 for x in Class_2])
    t = t[...,None]
    X = np.array([x for x in Class_1] + [x for x in Class_2])
    if plot :
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        ax.scatter(Class_1[:,0],Class_1[:,1],marker='D',c='b',linewidth=5)
        ax.scatter(Class_2[:,0],Class_2[:,1],marker='D',c='r',linewidth=5)
        plt.show()
    return X, t



µ0 = np.array([0,0])
S0 = np.array([[1,0],[0,1]])


µ_1 = np.array([-2,-2])
µ_2 = np.array([1,1])
S_1 = np.identity(2)
S_2 = 0.5*np.identity(2)
sample_size_1 = sample_size_2 = 30

X, t = generate_data(µ_1,µ_2,S_1,S_2,sample_size_1,sample_size_2,True)



# %%
def generate_basis_functions(m, µ_min_x1, µ_max_x1, µ_min_x2, µ_max_x2, S) :
    def constant(x) :
        return 1
    Set = [constant]
    for k_x1 in range(m) :
        for k_x2 in range(m) :
            µ = np.array([µ_min_x1 + k_x1*(µ_max_x1 - µ_min_x1)/m , µ_min_x2 + k_x2*(µ_max_x2 - µ_min_x2)/m])
            Set.append(Gaussian2D(µ,S))
    return Set

def phi(X, basis_functions) :
    N = len(X)
    M = len(basis_functions)
    phi = np.array([[basis_functions[j](X[i]) for j in range(M)] for i in range(N)])
    return phi



basis_functions = generate_basis_functions(10,-5,5,-5,5,0.5*np.identity(2))




# %%
def sigmoid(x) :
    if x >= 10 :
        return np.array([1 - 10**(-4)])
    if x <= -10 :
        return np.array([10**(-4)])
    return 1 / (1 + np.exp(-x))


def y(x, w, basis_functions) :
    phi_x = phi(x, basis_functions)[0]
    return sigmoid(phi_x@(w[...,None]))[0]

def Y(X,w,basis_functions) :
    return np.array([y(x,w,basis_functions) for x in X])

def classification(X,w,basis_functions) :
    return (Y(X,w,basis_functions) >= 0.5).astype(int)


def Newton_Raphson_update_w(w,X,t) :
    phi_X = phi(X,basis_functions)
    N = len(X)
    preds = Y(X,w,basis_functions)

    R = np.zeros([N,N])
    for i in range(N) :
        R[i,i] = preds[i] * (1 - preds[i])

    if  np.linalg.det(R)==0 :
        return w
    
    z = phi_X@(w[...,None]) - np.linalg.inv(R)@(preds[...,None] - t)
    return (np.linalg.inv(phi_X.T@R@phi_X)@phi_X.T@R@z).T[0]


#%%
w = np.array([np.random.uniform(-1,1) for i in basis_functions])
Data = [generate_data(µ_1,µ_2,S_1,S_2,sample_size_1,sample_size_2,False) for k in range(100)]
i = 0
w_new = Newton_Raphson_update_w(w,Data[0][0],Data[0][1])

for i in range(50) :
    compute_w = Newton_Raphson_update_w(w_new,Data[i][0],Data[i][1])
    w = w_new
    w_new = compute_w


x1 = np.random.uniform(-5,5,50)
x2 = np.random.uniform(-5,5,50)
X = [[x1[i],x2[i]] for i in range(len(x1))]
t = classification(X,w_new,basis_functions)
Class_1 = np.array([X[i] for i in range(len(X)) if t[i]==1])
Class_2 = np.array([X[i] for i in range(len(X)) if t[i]==0])
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.scatter(Class_1[:,0],Class_1[:,1],marker='D',c='b',linewidth=5)
ax.scatter(Class_2[:,0],Class_2[:,1],marker='D',c='r',linewidth=5)


# %%
X = np.array([[0,1],[1,0]])
w = np.array([np.random.uniform(-1,1) for i in basis_functions])
print(w.shape)
classification(X,w,basis_functions)
t = np.array([1,1])[...,None]
Newton_Raphson_update_w(w,X,t)