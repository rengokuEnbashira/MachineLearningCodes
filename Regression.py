import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import sys
import matplotlib.pyplot as plt 

if sys.argv[1] == "linear":
    # Using linear Regression
    x = np.linspace(0,5,100)
    y = 2*x - 5
    y += 5*np.random.random(y.shape)

    reg = LinearRegression()
    reg.fit(x.reshape(100,1),y)
    reg_data = reg.predict(x.reshape(100,1))
    print("Coefs",reg.coef_)
    print("Intercept",reg.intercept_)
    plt.scatter(x,y)
    plt.plot(x,reg_data,"g")
    plt.show()
elif sys.argv[1] == "nonlinear":
    # Using support vector regression
    x = np.linspace(0,5,100)
    y = 4*np.sin(2*x)
    y += 3*np.random.random(y.shape)
    reg = SVR(kernel="rbf",C=1000,gamma=0.1,epsilon=.1)
    reg.fit(x.reshape(100,1),y)
    reg_data = reg.predict(x.reshape(100,1))
    plt.scatter(x,y)
    plt.plot(x,reg_data)
    plt.show()
elif sys.argv[1] == "knn":
    # Using K nearest neighbors
    x = np.linspace(0,5,100)
    y = 4*np.sin(2*x)
    y += 3*np.random.random(y.shape)
    reg = KNeighborsRegressor()
    reg.fit(x.reshape(100,1),y)
    reg_data = reg.predict(x.reshape(100,1))
    plt.scatter(x,y)
    plt.plot(x,reg_data)
    plt.show()
