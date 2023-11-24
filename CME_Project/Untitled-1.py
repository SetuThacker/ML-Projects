# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# %%
df = pd.read_csv(r"D:\Coding\Learn_Python\Data\Training_Dataset_ME5201.csv", header=None)
df

# %%
X = df.iloc[1:,:-1]
y = df.iloc[1:,2:]

# %%
X

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
print(len(X_train),len(X_test))

# %%
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# %%
Lin_Reg = LinearRegression()
Lin_Reg.fit(X_train,y_train)

# %%
y_predict = Lin_Reg.predict(X_test)
y_predict

# %%
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
mse1= mean_squared_error(y_train,Lin_Reg.predict(X_train))
mae1= mean_absolute_error(y_train,Lin_Reg.predict(X_train))
raq1 = r2_score(y_train, Lin_Reg.predict(X_train))
print(mse1)
print(mae1)
print(raq1)

# %%
mse2= mean_squared_error(y_test,Lin_Reg.predict(X_test))
mae2= mean_absolute_error(y_test,Lin_Reg.predict(X_test))
raq2 = r2_score(y_test, Lin_Reg.predict(X_test))
print(mse2)
print(mae2)
print(raq2)

# %%
print(Lin_Reg.coef_, Lin_Reg.intercept_)

# %% [markdown]
# # Let's go  with Poly_Regnomial Regression

# %%
Poly_Reg = PolynomialFeatures(degree=3, include_bias=True)

X_train_trans = Poly_Reg.fit_transform(X_train)
X_test_trans = Poly_Reg.transform(X_test)

# %%
print(X_train[0])
print(X_train_trans[0])

# %%
Lin_Reg_2 = LinearRegression() 
Lin_Reg_2.fit(X_train_trans,y_train)

# %%
y_pred = Lin_Reg_2.predict(X_test_trans)

# %%
Lin_Reg_2.score(X_test_trans,y_pred)

# %%
r2_score(y_test,y_pred)

# %%
print(Lin_Reg_2.coef_)
print(Lin_Reg_2.intercept_)

# %% [markdown]
# # let's solve this using Nueral Network
# 

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
scalar = MinMaxScaler()
scalar.fit(X_train)

# %%
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# %%
pd.DataFrame(X_train).describe()

# %%
pd.DataFrame(X_test).describe()

# %%
from sklearn.neural_network import MLPRegressor
from  sklearn import metrics

# %%
nn = MLPRegressor(hidden_layer_sizes = (1000), activation= 'logistic', max_iter= 2000, solver= 'lbfgs')
nn.fit(X_train, y_train)

# %%
mae1 = metrics.mean_absolute_error(y_train, nn.predict(X_train))
mse1 = metrics.mean_squared_error(y_train, nn.predict(X_train))
raq1 = metrics.r2_score(y_train, nn.predict(X_train))
print(mae1, mse1, raq1)

# %%
mae2 = metrics.mean_absolute_error(y_test, nn.predict(X_test))
mse2 = metrics.mean_squared_error(y_test, nn.predict(X_test))
raq2 = metrics.r2_score(y_test, nn.predict(X_test))
print(mae2, mse2, raq2)

# %%
print(f"Number of inputs:  {nn.n_features_in_}")
print(f"Number of outputs: {nn.n_outputs_}")
print(f"Number of layers:  {nn.n_layers_}")
print(f"Layer sizes: {[l.shape for l in nn.coefs_]}")