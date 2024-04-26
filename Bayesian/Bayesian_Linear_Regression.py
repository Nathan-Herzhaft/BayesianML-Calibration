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
        exponent = (-1/2) * (x - µ)@np.linalg.inv(S)@np.transpose(x-µ)
        return ((1/Z) * np.exp(exponent))
    
    return f


def prediction(µ,S,x) :
    phi_x = np.array([[1,x]]).T
    mean = µ @ phi_x
    std = np.sqrt( sigma**2 +  phi_x.T[0]@S@phi_x)
    return mean[0], std[0]



#%%
####################################################################
#               Visualisations graphiques
####################################################################


def sample_models(µ,S,data_points=[],sample_size=5) :
    sample_coeff = np.random.multivariate_normal(µ,S,5)
    x = np.linspace(-1,1,20)
    fig,ax = plt.subplots(figsize=(8,8))
    ax.set_xlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylim([-1,1])
    ax.set_ylabel('Y')
    ax.set_title('Random sampling of 5 models according to the prior')
    for i in range(sample_size) :
        y =  sample_coeff[i,0] + sample_coeff[i,1]*x
        ax.plot(x,y,'r',lw=5)
    if len(data_points) != 0 :
        if len(data_points) == 1 :
            ax.scatter(data_points[0][0],data_points[0][1],marker='D',c = 'g',linewidth=5)
        else :
            ax.scatter(data_points[:,0],data_points[:,1],marker='D',c='g',linewidth=5)
    plt.show()

def Plot_Distrib(f) :
    X = Y = np.linspace(-1,1,21)
    Z = np.array([f(np.array([x,y])) for y in Y for x in X])
    Z = Z.reshape(21,21)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.contourf(X,Y,Z,100,cmap='jet')
    ax.set_xlim([-1,1])
    ax.set_xlabel('w0')
    ax.set_ylim([-1,1])
    ax.set_ylabel('w1')
    ax.set_title('Prior distribution : Density of probability    (y = w0 +w1*x)')
    plt.show()


def prediction_distribution_mean_and_standard_deviation(µ,S,infered_data) :
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylim([-1,1])
    ax.set_ylabel('Y')
    if len(infered_data) != 0 :
        if len(infered_data) == 1 :
            ax.scatter(infered_data[0][0],infered_data[0][1],marker='D',c='g',linewidth=5)
        else :
            ax.scatter(infered_data[:,0],infered_data[:,1],marker='D',c='g',linewidth=5)
    X = np.linspace(-1,1,20)
    mean = np.array([prediction(µ,S,x)[0] for x in X])
    std = np.array([prediction(µ,S,x)[1] for x in X])
    y1 = mean-std
    y2 = mean+std
    ax.plot(X,mean,'r')
    ax.set_title('Predictive distribution : Mean and standard deviation')
    ax.fill_between(X,y1,y2,color = 'r', alpha = 0.1)
    plt.show()


def prediction_distribution_density(µ,S,infered_data) : 
    X = Y = np.linspace(-1,1,21)
    Z = np.zeros((21,21))
    for i in range(len(X)) :
        mean, std = prediction(µ,S,X[i])
        f = Gaussian1D(mean,std)
        for j in range(len(Y)) :
            Z[j,i] = f(Y[j])
    fig, ax = plt.subplots(figsize=(8,8))
    ax.contourf(X,Y,Z,100,cmap='jet')
    if len(infered_data) != 0 :
        if len(infered_data) == 1 :
            ax.scatter(infered_data[0][0],infered_data[0][1],marker='D',c='g',linewidth=5)
        else :
            ax.scatter(infered_data[:,0],infered_data[:,1],marker='D',c='g',linewidth=5)
    ax.set_xlim([-1,1])
    ax.set_xlabel('X')
    ax.set_ylim([-1,1])
    ax.set_ylabel('Y')
    ax.set_title('Predictive distribution : Probability density')
    plt.show()






#%%
####################################################################
#           Initialisation du prior et des données
####################################################################

def generate_data(w0,w1,noise_std,sample_size) :
    noise = np.random.normal(0,noise_std,sample_size)
    X = np.random.uniform(-1,1,sample_size)
    Y = w0 + w1*X + noise
    Data = np.transpose(np.array([X, Y]))
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    x = np.linspace(-1,1,20)
    y = np.array(w1*x + w0)
    y1 = y - noise_std
    y2 = y + noise_std
    ax.plot(x,y,'--r')
    ax.set_title('Dataset Visualisation')
    ax.fill_between(x,y1,y2,color = 'r', alpha = 0.1)
    ax.scatter(Data[:,0],Data[:,1],marker='D',c='g',linewidth=5)
    ax.legend(('Objective function','Noise : Standard deviation','Sampling of data'))
    plt.show()
    return Data


def init(w0,w1,sigma,sample_size,plot_distrib_prior=True,plot_sample_models=True) :
    Data = generate_data(w0,w1,sigma,sample_size)
    µ = np.array([0,0])
    S = np.array([[1,0],[0,1]])

    if plot_distrib_prior :
        f = Gaussian2D(µ,S)
        Plot_Distrib(f)
    if plot_sample_models :
        sample_models(µ,S)
    return µ, S, sigma, Data




#%%
####################################################################
#                   Fonctions d'entraînements
####################################################################


def compute_posterior_one_step(µ,S,sigma,data_point) :
    beta = 1/(sigma**2)
    µ = µ[..., None]
    phi = np.array([[1,data_point[0]]])
    new_S = np.linalg.inv(np.linalg.inv(S) + beta*(phi.T)@phi)
    new_µ = new_S@(np.linalg.inv(S)@µ + beta*data_point[1]*(phi.T))
    new_µ = (new_µ.T)[0]
    return new_µ, new_S


def infer_one_sample(µ,S,sigma,Data,i,infered_data,plot_prior_density=True,plot_sample_models=True,plot_predict_mean_std=True,plot_prediction_density=True) :
    sample = Data[i]
    µ,S = compute_posterior_one_step(µ,S,sigma,sample)
    infered_data.append(sample)
    if plot_prior_density :
        f = Gaussian2D(µ,S)
        Plot_Distrib(f)
    if plot_sample_models :
        sample_models(µ,S,np.array(infered_data))
    if plot_predict_mean_std :
        prediction_distribution_mean_and_standard_deviation(µ,S,np.array(infered_data))
    if plot_prediction_density :
        prediction_distribution_density(µ,S,np.array(infered_data))
    return µ,S, infered_data

def infer_step_by_step(µ,S,sigma,Data,n_sample,n_plot) :
    infered_data = []
    for n in range(n_sample) :
        if (n_sample < len(Data)) :
            if (n_plot !=0 and n%n_plot == 0)or(n == n_sample -1) :
                print(f'Number of infered points : {n}')
                µ, S, infered_data = infer_one_sample(µ,S,sigma,Data,n,infered_data,True,True,True,True)
            else :
                µ, S, infered_data = infer_one_sample(µ,S,sigma,Data,n,infered_data,False,False,False,False)
        else :
            print('Not enough data points')
            return µ,S, np.array(infered_data)
    return µ,S, infered_data


def compute_posterior(µ,S,sigma,Data) :
    beta = 1/(sigma**2)
    µ = µ[..., None]
    X = Data[:,0]
    phi = np.array([[1,x] for x in X])
    X = X[...,None]
    y = Data[:,1]
    y = y[...,None]
    new_S = np.linalg.inv(np.linalg.inv(S) + beta*(phi.T)@phi)
    new_µ = new_S@(np.linalg.inv(S)@µ + beta*(phi.T)@y)
    new_µ = (new_µ.T)[0]
    return new_µ, new_S

def infer(µ,S,sigma,Data,n_sample,plot_prior_density=True,plot_sample_models=True,plot_predict_mean_std=True,plot_prediction_density=True) :
    Data = Data[:n_sample]
    µ,S = compute_posterior(µ,S,sigma,Data)
    print('Training complete')
    print(f'Number of infered points : {n_sample}')
    if plot_prior_density :
        f = Gaussian2D(µ,S)
        Plot_Distrib(f)
    if plot_predict_mean_std :
        prediction_distribution_mean_and_standard_deviation(µ,S,Data)
    if plot_sample_models :
        sample_models(µ,S,Data)
    if plot_prediction_density :
        prediction_distribution_density(µ,S,Data)
    return µ,S   





# %%
µ, S, sigma, Data = init(-0.3,0.5,0.2,50)
µ,S = infer(µ,S,sigma,Data,4)


#%%
µ, S, sigma, Data = init(-0.3,0.5,0.2,50)
µ,S,infered_data = infer_step_by_step(µ,S,sigma,Data,20,3)

# %%
