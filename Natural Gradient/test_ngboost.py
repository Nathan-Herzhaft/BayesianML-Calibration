#%%
from ngboost import NGBClassifier
from ngboost.distns import k_categorical,Bernoulli
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# %%
X, y  = load_breast_cancer(True)
X_train, X_test, y_train, y_test = train_test_split(X,y)

model = NGBClassifier(Dist=Bernoulli,verbose=True)
# %%
model.fit(X_train,y_train)
# %%
model.predict(X_test)[0:5]
# %%
model.pred_dist(X_test)[0:5].params
# %%
