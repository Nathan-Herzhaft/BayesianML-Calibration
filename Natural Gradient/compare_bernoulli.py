#%%
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import norm
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# %%
############################################################
#                          Data
############################################################

X = np.random.normal(0,1,10000)[...,None]
norm_cdf = norm.cdf(X)
y = np.array([np.random.binomial(1,p) for p in norm_cdf])

X_train, X_test, y_train, y_test = train_test_split(X,y)


# %%
############################################################
#                       XGBoost
############################################################

model = XGBClassifier(learning_rate=0.1,
                      n_estimators=100,
                      max_depth=3,
                      )

eval_set = [(X_test,y_test)]
model.fit(X_train,
          y_train,
          eval_set=eval_set,
          early_stopping_rounds=3,
          )


probs = model.predict_proba(X_test)[:,1]
plt.hist(probs,bins=10)


# %%
############################################################
#                       NGBoost
############################################################

tree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)

model = NGBClassifier(Dist=Bernoulli,
                      learning_rate=0.1,
                      n_estimators=100,
                      Base=tree,
                      verbose=True)

model.fit(X_train,y_train,
          X_val=X_test,
          Y_val=y_test,
          early_stopping_rounds=3
          )

probs = model.pred_dist(X_test).params['p1']
plt.hist(probs,bins=10)




# %%
