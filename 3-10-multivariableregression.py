import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

# y = a0 + a1x1 + a2x2 + ... + anxn multivariable 

X = np.random.rand(100,2)
coef = np.array([3,5])
y = 0 + np.dot(X,coef)


lin_reg = LinearRegression()
lin_reg.fit(X, y)


fig = plt.figure()
ax = fig.add_subplot(111,projection = "3d")
ax.scatter(X[:,0],X[:,1],y)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_ylabel("y")

x1, x2 = np.mashgrid(np.linspace(0,1,10),np.linspace(0,1,10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1, x2 , y_pred.reshape(x1. shape))
plt.title("multi variable linear regression")

print("katsayilar: ",lin_reg.coef_)
print("kesim:",lin_reg.intercept_)


#%%

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()

X = diabetes.data 
y = diabetes.target

X_train ,X_test ,y_train , y_test = train_test_split(X , y ,test_size=0.2 , random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

rmse = mean_squared_error (y_test, y_pred, squared=False)
print("rmse:", rmse)
