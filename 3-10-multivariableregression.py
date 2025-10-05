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




