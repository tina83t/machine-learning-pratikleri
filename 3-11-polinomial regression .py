import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = 4 * np.random.rand(100,1)
y = 2 + 3 * X ** 2

plt.scatter(X, y)

poly_feat = PolynomialFeatures(degree= 2)
X_poly = poly_feat.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

plt.scatter(X, y, color = "blue")

X_test = np.linspace(0,4,100).reshape(-1,1)
X_test_poly = poly_feat.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

plt.plot(X_test, y_pred, color = "red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("polinom regression modeli")


#%%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()

X = housing.data 
y = housing.target

X_train ,X_test ,y_train ,y_test = train_test_split(X, y ,test_size=0.2, random_state=42)

poly_feat = PolynomialFeatures(degree=2)
X_train_poly = poly_feat.fit_transform(X_train)
X_test_poly = poly_feat.fit_transform(X_train)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred = poly_reg.predict(X_test_poly)

print("polunomial regression rmse:",mean_squared_error(y_test, y_pred, squared = False))

lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)
y_pred = lin_reg.predict(X_test)

print("multi variable linear regression rmse:",mean_squared_error(y_test, y_pred, squared = False))





